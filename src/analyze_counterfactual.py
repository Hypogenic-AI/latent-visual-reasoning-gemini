
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from src.modules import SlotVideoModel
import os

def analyze_counterfactual():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Collision Video
    video = torch.load('datasets/collision/collision.pt').to(device) # (1, T, C, H, W)
    
    model = SlotVideoModel(num_slots=6, slot_dim=64).to(device)
    model.load_state_dict(torch.load('results/models/slot_model.pth'))
    model.eval()
    
    # 1. Encode to get slots
    input_seq = video[:, :-1]
    T = input_seq.shape[1]
    
    # Run full forward pass to get slots and masks
    with torch.no_grad():
        out = model(input_seq)
        
    slots = out['slots'] # (1, T, N, D)
    masks = out['masks'] # (1, T, N, 1, H, W)
    
    # 2. Identify Red Ball Slot
    # At t=0, Red Ball is at specific position.
    # But easier: Use the mask color.
    # Red is channel 0.
    
    # Let's look at masks at t=0
    masks_0 = masks[0, 0] # (N, 1, H, W)
    img_0 = input_seq[0, 0] # (3, H, W)
    
    # Find slot with highest overlap with Red channel
    red_score = []
    for n in range(6):
        # Weighted sum of red channel
        mask = masks_0[n, 0]
        score = (mask * img_0[0]).sum()
        red_score.append(score.item())
        
    red_idx = np.argmax(red_score)
    print(f"Identified Red Ball as Slot {red_idx}")
    
    # 3. Counterfactual Simulation
    # We want to remove Red Slot at t=0 and roll out dynamics.
    
    # Initial state
    curr_slots = slots[:, 0] # (1, N, D)
    
    # Remove Red Slot (Zero out)
    cf_slots = curr_slots.clone()
    cf_slots[:, red_idx] = 0
    
    # Autoregressive Rollout
    # We need to rollout both Original (for ref) and CF
    
    orig_curr = curr_slots
    cf_curr = cf_slots
    
    orig_frames = []
    cf_frames = []
    
    with torch.no_grad():
        for t in range(T):
            # Original Step
            interacted = model.predictor.transformer(orig_curr)
            delta = model.predictor.predictor(interacted)
            orig_next = orig_curr + delta
            
            # Decode Orig
            orig_img, _, _, _ = model.decoder(orig_next)
            orig_img = orig_img.reshape(1, 3, 64, 64)
            orig_frames.append(orig_img)
            
            orig_curr = orig_next
            
            # CF Step
            # IMPORTANT: We must enforce "Removed Slot stays Removed" or hope dynamics handles it?
            # Ideally, if slot is 0, it stays 0? Or dynamics might "revive" it?
            # Let's force zero it at every step to be sure we are testing "Interaction with Ghost".
            # Or better: let it evolve. If it stays 0, model is stable.
            # But "0" might not be the "empty" embedding.
            # Let's just Zero it out before interaction step to simulate "It's gone".
            
            cf_curr[:, red_idx] = 0
            
            interacted_cf = model.predictor.transformer(cf_curr)
            delta_cf = model.predictor.predictor(interacted_cf)
            cf_next = cf_curr + delta_cf
            
            # Decode CF
            cf_img, _, _, _ = model.decoder(cf_next)
            cf_img = cf_img.reshape(1, 3, 64, 64)
            cf_frames.append(cf_img)
            
            cf_curr = cf_next
            
    # Create GIF
    # Top: Ground Truth
    # Middle: Original Pred
    # Bottom: Counterfactual Pred (Red Removed)
    
    gif_frames = []
    for t in range(len(orig_frames)):
        gt = video[0, t+1].permute(1, 2, 0).cpu().numpy()
        orig = orig_frames[t][0].permute(1, 2, 0).cpu().numpy()
        cf = cf_frames[t][0].permute(1, 2, 0).cpu().numpy()
        
        gt = np.clip(gt, 0, 1) * 255
        orig = np.clip(orig, 0, 1) * 255
        cf = np.clip(cf, 0, 1) * 255
        
        sep = np.ones((2, 64, 3)) * 255
        combined = np.concatenate([gt, sep, orig, sep, cf], axis=0)
        gif_frames.append(combined.astype(np.uint8))
        
    imageio.mimsave('results/plots/counterfactual.gif', gif_frames, fps=4)
    print("Counterfactual GIF saved to results/plots/counterfactual.gif")

if __name__ == '__main__':
    analyze_counterfactual()
