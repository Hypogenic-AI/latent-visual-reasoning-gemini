
import torch
import imageio
import numpy as np
import os
from src.modules import SlotVideoModel
from src.baseline import ConvPredictor

def create_gif():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    test_data = torch.load('datasets/bouncing_balls/test.pt')
    
    # Init Models
    slot_model = SlotVideoModel(num_slots=6, slot_dim=64).to(device)
    baseline_model = ConvPredictor(hidden_dim=32).to(device)
    
    # Load Weights
    slot_model.load_state_dict(torch.load('results/models/slot_model.pth'))
    baseline_model.load_state_dict(torch.load('results/models/baseline_model.pth'))
    
    slot_model.eval()
    baseline_model.eval()
    
    # Select a good example
    # We'll take the first one from test set
    idx = 0
    video = test_data[idx:idx+1].to(device) # (1, T, C, H, W)
    
    input_seq = video[:, :-1]
    target_seq = video[:, 1:] # This is what we want to match (Frame 1 to T)
    
    with torch.no_grad():
        out_slot = slot_model(input_seq)
        out_base = baseline_model(input_seq)
        
    pred_slot = out_slot['pred_next_img'] # (1, T-1, 3, H, W)
    pred_base = out_base # (1, T-1, 3, H, W)
    
    # We want to visualize the sequence:
    # Top: Ground Truth (Frame 1...T)
    # Middle: Baseline Pred (Frame 1...T)
    # Bottom: Slot Pred (Frame 1...T)
    
    frames = []
    T = target_seq.shape[1]
    
    for t in range(T):
        # Get frames, permute to HWC, convert to numpy, scale to 0-255
        gt = target_seq[0, t].permute(1, 2, 0).cpu().numpy()
        base = pred_base[0, t].permute(1, 2, 0).cpu().numpy()
        slot = pred_slot[0, t].permute(1, 2, 0).cpu().numpy()
        
        # Clip and Scale
        gt = np.clip(gt, 0, 1) * 255
        base = np.clip(base, 0, 1) * 255
        slot = np.clip(slot, 0, 1) * 255
        
        # Concatenate vertically
        # Add a white separator line
        sep = np.ones((2, 64, 3)) * 255
        frame_combined = np.concatenate([gt, sep, base, sep, slot], axis=0)
        
        frames.append(frame_combined.astype(np.uint8))
        
    save_path = 'results/plots/comparison.gif'
    imageio.mimsave(save_path, frames, fps=4)
    print(f"GIF saved to {save_path}")

if __name__ == '__main__':
    create_gif()
