
import torch
import torch.nn as nn

class ConvPredictor(nn.Module):
    """
    Baseline: Unstructured Video Prediction.
    Frame_t -> ConvEncoder -> ConvDecoder -> Frame_{t+1}
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=4, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=4, stride=2, padding=1), # 8x8
            nn.ReLU()
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dim*4, hidden_dim*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim*4, hidden_dim*4, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim*2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Pixel values 0-1
        )
        
    def forward(self, video):
        # video: (B, T, C, H, W)
        b, t, c, h, w = video.shape
        
        # Flatten to (B*T, C, H, W)
        flat_video = video.reshape(b*t, c, h, w)
        
        enc = self.encoder(flat_video)
        bot = self.bottleneck(enc)
        pred = self.decoder(bot)
        
        pred = pred.reshape(b, t, c, h, w)
        return pred
