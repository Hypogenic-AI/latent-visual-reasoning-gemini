
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)
        slots = mu + sigma * torch.randn(mu.shape, device = device, dtype = dtype)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True) # Weighted mean

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots

class Encoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2, stride=2), # 32x32
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2, stride=2), # 16x16
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=5, padding=2)
        )
        self.pos_emb = nn.Parameter(torch.randn(1, 16*16, hidden_dim) * 0.02)
        
    def forward(self, x):
        b = x.shape[0]
        x = self.cnn(x)
        # Permute to (B, H*W, C)
        x = x.permute(0, 2, 3, 1).reshape(b, -1, x.shape[1])
        return x + self.pos_emb

class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, hidden_dim, resolution=(64, 64)):
        super().__init__()
        self.resolution = resolution
        self.hidden_dim = hidden_dim
        
        # Positional embedding for decoder
        x = torch.linspace(-1, 1, resolution[0])
        y = torch.linspace(-1, 1, resolution[1])
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        pos_grid = torch.stack((xx, yy), dim=0).unsqueeze(0) # (1, 2, H, W)
        self.register_buffer('pos_grid', pos_grid)
        
        self.decoder_cnn = nn.Sequential(
            nn.Conv2d(hidden_dim + 2, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 4, kernel_size=3, padding=1) # RGB + Alpha
        )
        
    def forward(self, slots):
        # slots: (B, N, D)
        b, n, d = slots.shape
        h, w = self.resolution
        
        # Broadcast slots to spatial grid
        slots = slots.reshape(b * n, d, 1, 1).expand(-1, -1, h, w)
        
        # Add positional embedding
        pos = self.pos_grid.expand(b * n, -1, -1, -1)
        x = torch.cat((slots, pos), dim=1)
        
        # Decode
        x = self.decoder_cnn(x) # (B*N, 4, H, W)
        
        # Unpack
        x = x.reshape(b, n, 4, h, w)
        recons = x[:, :, :3]
        masks = F.softmax(x[:, :, 3], dim=1).unsqueeze(2) # Softmax over slots for alpha
        
        # Reconstruct image
        image = torch.sum(recons * masks, dim=1)
        return image, recons, masks, x[:, :, 3]

class SlotPredictor(nn.Module):
    """
    Predicts next slot states given history.
    Treats each slot as an independent entity but allows interaction via Transformer.
    """
    def __init__(self, slot_dim, history_len=5, use_interaction=True):
        super().__init__()
        self.use_interaction = use_interaction
        self.history_len = history_len
        
        if self.use_interaction:
            # Custom Transformer Block to access weights
            self.attn = nn.MultiheadAttention(embed_dim=slot_dim, num_heads=4, batch_first=True)
            self.norm1 = nn.LayerNorm(slot_dim)
            self.ff = nn.Sequential(
                nn.Linear(slot_dim, slot_dim*4),
                nn.ReLU(),
                nn.Linear(slot_dim*4, slot_dim)
            )
            self.norm2 = nn.LayerNorm(slot_dim)
        
        self.predictor = nn.Linear(slot_dim, slot_dim) # Simple transition
        
    def forward(self, slot_history):
        # slot_history: (B, T, N, D) or (B, N, D)
        
        if len(slot_history.shape) == 4:
            current_slots = slot_history[:, -1] # (B, N, D)
        else:
            current_slots = slot_history
            
        weights = None
        
        if self.use_interaction:
            # Self-Attention
            attn_out, weights = self.attn(current_slots, current_slots, current_slots)
            x = self.norm1(current_slots + attn_out)
            # Feed Forward
            ff_out = self.ff(x)
            interacted = self.norm2(x + ff_out)
        else:
            interacted = current_slots
            
        # Dynamics step
        delta = self.predictor(interacted)
        next_slots = current_slots + delta
        return next_slots, weights

class SlotVideoModel(nn.Module):
    def __init__(self, resolution=(64, 64), num_slots=4, slot_dim=64, use_interaction=True):
        super().__init__()
        self.encoder = Encoder(hidden_dim=slot_dim)
        self.slot_attention = SlotAttention(num_slots=num_slots, dim=slot_dim, iters=3, hidden_dim=slot_dim*2)
        self.decoder = SpatialBroadcastDecoder(hidden_dim=slot_dim, resolution=resolution)
        self.predictor = SlotPredictor(slot_dim=slot_dim, use_interaction=use_interaction)
        self.use_interaction = use_interaction
        
    def forward(self, video, burn_in=5):
        # video: (B, T, C, H, W)
        b, t, c, h, w = video.shape
        
        # 1. Encode all frames to slots
        # Reshape to (B*T, C, H, W) to process in parallel
        flat_video = video.reshape(b * t, c, h, w)
        features = self.encoder(flat_video)
        slots = self.slot_attention(features) # (B*T, N, D)
        slots = slots.reshape(b, t, -1, slots.shape[-1]) # (B, T, N, D)
        
        # 2. Predict future slots
        
        # Parallel prediction
        flat_slots = slots.reshape(b * t, -1, slots.shape[-1])
        
        # Predictor now returns (next_slots, weights)
        interacted, weights = self.predictor(flat_slots)
        
        # interacted IS next_slots in the new implementation (see above: next_slots = current + delta)
        # In previous code: delta = predictor(interacted).
        # In new code: next_slots = current + delta.
        # So 'interacted' variable here actually holds 'next_slots'.
        
        next_slots_pred = interacted.reshape(b, t, -1, slots.shape[-1])
        
        if weights is not None:
            weights = weights.reshape(b, t, -1, slots.shape[-2]) # (B, T, N, N)
        
        # 3. Decode predicted slots
        
        flat_next_slots = next_slots_pred.reshape(b*t, -1, slots.shape[-1])
        pred_next_img, _, _, _ = self.decoder(flat_next_slots)
        pred_next_img = pred_next_img.reshape(b, t, 3, h, w)
        
        # Also decode original slots
        recon_img, _, masks, _ = self.decoder(flat_slots)
        recon_img = recon_img.reshape(b, t, 3, h, w)
        masks = masks.reshape(b, t, -1, 1, h, w)
        
        return {
            'pred_next_img': pred_next_img, # Predictions for t+1
            'recon_img': recon_img,         # Reconstructions for t
            'masks': masks,
            'slots': slots,
            'next_slots_pred': next_slots_pred,
            'attn_weights': weights # (B, T, N, N)
        }
