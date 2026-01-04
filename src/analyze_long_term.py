
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from src.modules import SlotVideoModel
from src.baseline import ConvPredictor
from tqdm import tqdm

def analyze_long_term():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    test_data = torch.load('datasets/bouncing_balls/test.pt')
    
    # Init Models
    slot_model = SlotVideoModel(num_slots=6, slot_dim=64).to(device)
    baseline_model = ConvPredictor(hidden_dim=32).to(device)
    
    slot_model.load_state_dict(torch.load('results/models/slot_model.pth'))
    baseline_model.load_state_dict(torch.load('results/models/baseline_model.pth'))
    
    slot_model.eval()
    baseline_model.eval()
    
    criterion = nn.MSELoss(reduction='none')
    
    # We will test on 20 videos
    num_videos = 20
    test_batch = test_data[:num_videos].to(device) # (B, T, C, H, W)
    
    B, T, C, H, W = test_batch.shape
    # We feed Frame 0. We want to predict Frame 1..T-1.
    # We do this autoregressively.
    
    slot_preds = [] # List of (B, C, H, W) for each step
    base_preds = []
    
    # Initial input: Frame 0
    # Note: Models expect sequence input.
    # Slot Model: Encoder takes (B, T, ...)
    # But for autoregressive, we have state.
    
    # Slot Model Architecture Recap:
    # 1. Video -> Encoder -> Slots (B, T, N, D)
    # 2. Predictor: Slots(t) -> Slots(t+1)
    # 3. Decoder: Slots(t+1) -> Image(t+1)
    
    # Autoregressive Strategy for Slot Model:
    # 1. Encode Frame 0 -> Slots_0
    # 2. Loop t=0 to T-2:
    #    Slots_{t+1} = Predictor(Slots_t)
    #    Image_{t+1} = Decoder(Slots_{t+1})
    #    Store Image_{t+1}
    #    Update Slots_t = Slots_{t+1}
    
    # Autoregressive Strategy for Baseline:
    # Baseline takes (B, T, C, H, W) and outputs (B, T, ...). 
    # It's trained as seq-to-seq.
    # But technically it's ConvPredictor: Frame_t -> Frame_{t+1} (independent of history in my implementation? Let's check baseline.py)
    
    # Checking baseline.py:
    # forward(video): flattens B*T -> Encoder -> Decoder -> Reshape.
    # Yes, it processes each frame independently! It has NO memory.
    # So Frame_{t+1} depends ONLY on Frame_t.
    
    print("Running Autoregressive Generation...")
    
    with torch.no_grad():
        # --- Slot Model ---
        # 1. Get initial slots from Frame 0
        frame_0 = test_batch[:, 0:1] # (B, 1, C, H, W)
        flat_frame_0 = frame_0.reshape(B, C, H, W)
        features_0 = slot_model.encoder(flat_frame_0)
        current_slots = slot_model.slot_attention(features_0) # (B, N, D)
        
        slot_gen_frames = []
        
        curr_s = current_slots
        for t in range(T-1):
            # Predict next slots
            # Predictor takes (B, N, D) -> (B, N, D)?
            # In modules.py: Predictor.forward takes (B, T, N, D) and uses [:, -1]
            # Let's check modules.py predictor signature.
            # forward(slot_history): current = slot_history[:, -1]...
            # So we can pass (B, 1, N, D)
            
            interacted = slot_model.predictor.transformer(curr_s) # (B, N, D)
            delta = slot_model.predictor.predictor(interacted)
            next_s = curr_s + delta
            
            # Decode
            # Decoder expects (B, N, D)
            pred_img, _, _, _ = slot_model.decoder(next_s)
            pred_img = pred_img.reshape(B, 3, H, W)
            
            slot_gen_frames.append(pred_img)
            
            curr_s = next_s # Autoregress
            
        slot_gen_seq = torch.stack(slot_gen_frames, dim=1) # (B, T-1, C, H, W)
        
        # --- Baseline Model ---
        # Frame 0 -> Frame 1 -> Frame 2...
        curr_img = test_batch[:, 0:1] # (B, 1, C, H, W)
        
        base_gen_frames = []
        
        for t in range(T-1):
            # Baseline forward takes (B, T, ...) returns (B, T, ...)
            # We pass sequence length 1
            next_img = baseline_model(curr_img) # (B, 1, C, H, W)
            base_gen_frames.append(next_img.squeeze(1))
            
            curr_img = next_img # Autoregress
            
        base_gen_seq = torch.stack(base_gen_frames, dim=1) # (B, T-1, C, H, W)
        
    # Calculate MSE over time
    # Target: Frame 1 to T-1
    target_seq = test_batch[:, 1:]
    
    mse_slot_t = ((slot_gen_seq - target_seq)**2).mean(dim=(0, 2, 3, 4)).cpu().numpy()
    mse_base_t = ((base_gen_seq - target_seq)**2).mean(dim=(0, 2, 3, 4)).cpu().numpy()
    
    # Plot
    time_steps = np.arange(1, T)
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, mse_slot_t, '-o', label='Slot Dynamics (Ours)')
    plt.plot(time_steps, mse_base_t, '-x', label='Baseline (Conv)')
    plt.xlabel('Prediction Step (Frame into future)')
    plt.ylabel('MSE')
    plt.title('Long-Term Prediction Error Accumulation')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/long_term_error.png')
    print("Long term error plot saved to results/plots/long_term_error.png")
    
    # Visualize divergence at t=5, t=10, t=15
    # For the first video
    fig, axs = plt.subplots(3, 4, figsize=(12, 9))
    # Rows: t=5, 10, 15
    # Cols: Ground Truth, Baseline, Slot
    
    indices = [4, 9, 14] # 0-indexed, corresponding to Frame 5, 10, 15
    
    for i, idx in enumerate(indices):
        t_label = idx + 1
        
        # GT
        gt_img = target_seq[0, idx].permute(1, 2, 0).cpu().numpy()
        axs[i, 0].imshow(np.clip(gt_img, 0, 1))
        axs[i, 0].set_title(f"GT Frame {t_label}")
        axs[i, 0].axis('off')
        
        # Baseline
        base_img = base_gen_seq[0, idx].permute(1, 2, 0).cpu().numpy()
        axs[i, 1].imshow(np.clip(base_img, 0, 1))
        axs[i, 1].set_title(f"Base Frame {t_label}")
        axs[i, 1].axis('off')
        
        # Slot
        slot_img = slot_gen_seq[0, idx].permute(1, 2, 0).cpu().numpy()
        axs[i, 2].imshow(np.clip(slot_img, 0, 1))
        axs[i, 2].set_title(f"Slot Frame {t_label}")
        axs[i, 2].axis('off')
        
        # Slot Mask (to see if slots persist)
        # We need to decode masks for this slot state
        # curr_s was lost in loop, but we can re-run decoder if we stored slots
        # For now, just show images.
        axs[i, 3].axis('off') # Placeholder or remove column
        
    plt.tight_layout()
    plt.savefig('results/plots/long_term_viz.png')
    print("Long term viz saved to results/plots/long_term_viz.png")

if __name__ == '__main__':
    analyze_long_term()
