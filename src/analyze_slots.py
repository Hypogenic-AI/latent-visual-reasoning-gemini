
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.modules import SlotVideoModel
import os

def get_center_of_mass(masks):
    """
    masks: (T, N, 1, H, W)
    Returns: (T, N, 2) coordinates (y, x)
    """
    T, N, _, H, W = masks.shape
    device = masks.device
    
    # Create grid
    y = torch.arange(H, device=device).float()
    x = torch.arange(W, device=device).float()
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij') # (H, W)
    
    # Normalize masks to sum to 1 (soft attention)
    # masks are already softmaxed over N, but here we want spatial COM for each slot
    # So we treat the mask image as a density map
    
    # Problem: masks are soft alpha.
    # We want COM of the mask.
    
    masks_flat = masks.view(T, N, -1)
    mass = masks_flat.sum(dim=-1, keepdim=True) + 1e-6
    
    # COM y
    weighted_y = (masks * grid_y.view(1, 1, 1, H, W)).view(T, N, -1).sum(dim=-1, keepdim=True)
    com_y = weighted_y / mass
    
    # COM x
    weighted_x = (masks * grid_x.view(1, 1, 1, H, W)).view(T, N, -1).sum(dim=-1, keepdim=True)
    com_x = weighted_x / mass
    
    return torch.cat([com_y, com_x], dim=-1).squeeze(-1) # (T, N, 2)

def analyze_slots():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    test_data = torch.load('datasets/bouncing_balls/test.pt')
    
    model = SlotVideoModel(num_slots=6, slot_dim=64).to(device)
    model.load_state_dict(torch.load('results/models/slot_model.pth'))
    model.eval()
    
    # Analyze first 5 videos
    num_videos = 5
    video_batch = test_data[:num_videos].to(device)
    input_seq = video_batch[:, :-1]
    
    with torch.no_grad():
        out = model(input_seq)
    
    # out['masks']: (B, T, N, 1, H, W)
    masks = out['masks']
    
    # Plot trajectories for the first video
    b_idx = 0
    vid_masks = masks[b_idx] # (T, N, 1, H, W)
    
    # Calculate mass to filter empty slots
    # vid_masks: (T, N, 1, H, W)
    slot_mass = vid_masks.sum(dim=(0, 2, 3, 4)) # Sum over time and space for rough check?
    # Better: check mass per timestep
    
    mass_per_step = vid_masks.sum(dim=(2, 3, 4)) # (T, N)
    avg_mass = mass_per_step.mean(dim=0).cpu().numpy() # (N)
    
    print(f"Slot Average Masses: {avg_mass}")
    
    coms = get_center_of_mass(vid_masks) # (T, N, 2)
    coms = coms.cpu().numpy()
    
    plt.figure(figsize=(8, 8))
    plt.title("Slot Trajectories (Center of Mass)")
    plt.xlim(0, 64)
    plt.ylim(64, 0) # Image coordinates
    
    colors = plt.cm.jet(np.linspace(0, 1, 6))
    
    valid_slots = []
    
    for n in range(6):
        if avg_mass[n] < 5.0: # Threshold for "empty" slot
            continue
            
        valid_slots.append(n)
        
        # Plot path
        y = coms[:, n, 0]
        x = coms[:, n, 1]
        
        plt.plot(x, y, '-o', label=f'Slot {n}', color=colors[n], markersize=4, alpha=0.7)
        # Mark start
        plt.plot(x[0], y[0], 'x', color=colors[n], markersize=8)
        
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/slot_trajectories.png')
    print("Trajectory plot saved to results/plots/slot_trajectories.png")
    
    # Calculate Velocity Consistency for VALID slots
    if len(valid_slots) > 0:
        valid_coms = coms[:, valid_slots, :] # (T, N_valid, 2)
        velocities = valid_coms[1:] - valid_coms[:-1]
        accelerations = velocities[1:] - velocities[:-1]
        
        mean_acc = np.mean(np.linalg.norm(accelerations, axis=-1))
        print(f"Mean Slot Acceleration (Valid Slots Only): {mean_acc:.4f}")
    else:
        mean_acc = 0.0
        print("No valid slots found.")
    
    # Save metric
    with open('results/slot_metrics.txt', 'w') as f:
        f.write(f"Mean Slot Acceleration: {mean_acc:.4f}\n")
        f.write(f"Valid Slots: {valid_slots}\n")
        
    # Visualize Masks for t=0
    # masks: (B, T, N, 1, H, W)
    masks_t0 = vid_masks[0] # (N, 1, H, W)
    
    plt.figure(figsize=(12, 2))
    for n in range(6):
        plt.subplot(1, 6, n+1)
        plt.imshow(masks_t0[n, 0].cpu().numpy(), vmin=0, vmax=1, cmap='gray')
        plt.title(f"Slot {n}\nMass: {avg_mass[n]:.1f}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/plots/slot_masks_debug.png')
    print("Mask debug plot saved to results/plots/slot_masks_debug.png")

if __name__ == '__main__':
    analyze_slots()
