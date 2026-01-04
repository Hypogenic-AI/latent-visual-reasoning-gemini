
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.modules import SlotVideoModel
import os

def visualize_ood_failure():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load OOD Data (6 balls)
    test_data = torch.load('datasets/bouncing_balls_ood/test_6balls.pt')
    
    model = SlotVideoModel(num_slots=6, slot_dim=64).to(device)
    model.load_state_dict(torch.load('results/models/slot_model.pth'))
    model.eval()
    
    # Take one video
    video = test_data[0:1].to(device)
    input_seq = video[:, :-1]
    
    with torch.no_grad():
        out = model(input_seq)
        
    # out['pred_next_img']: (1, T, 3, H, W)
    # out['masks']: (1, T, N, 1, H, W)
    
    pred = out['pred_next_img'][0].cpu().numpy()
    masks = out['masks'][0].cpu().numpy() # (T, N, 1, H, W)
    gt = video[0, 1:].cpu().numpy()
    
    # Visualize Frame 10 (mid-sequence)
    t = 10
    
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: GT, Pred
    axs[0, 0].imshow(np.clip(gt[t].transpose(1, 2, 0), 0, 1))
    axs[0, 0].set_title("Ground Truth (6 Balls)")
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(np.clip(pred[t].transpose(1, 2, 0), 0, 1))
    axs[0, 1].set_title("Prediction")
    axs[0, 1].axis('off')
    
    # Visualize Masks (First 6 slots)
    # We want to see if slots split objects or merge them.
    # We'll plot the top 6 slots (all of them).
    
    # Row 1, Col 2-3: Masks 0 and 1
    axs[0, 2].imshow(masks[t, 0, 0], cmap='gray', vmin=0, vmax=1)
    axs[0, 2].set_title("Slot 0 Mask")
    axs[0, 2].axis('off')
    
    axs[0, 3].imshow(masks[t, 1, 0], cmap='gray', vmin=0, vmax=1)
    axs[0, 3].set_title("Slot 1 Mask")
    axs[0, 3].axis('off')
    
    # Row 2: Masks 2-5
    axs[1, 0].imshow(masks[t, 2, 0], cmap='gray', vmin=0, vmax=1)
    axs[1, 0].set_title("Slot 2 Mask")
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(masks[t, 3, 0], cmap='gray', vmin=0, vmax=1)
    axs[1, 1].set_title("Slot 3 Mask")
    axs[1, 1].axis('off')
    
    axs[1, 2].imshow(masks[t, 4, 0], cmap='gray', vmin=0, vmax=1)
    axs[1, 2].set_title("Slot 4 Mask")
    axs[1, 2].axis('off')
    
    axs[1, 3].imshow(masks[t, 5, 0], cmap='gray', vmin=0, vmax=1)
    axs[1, 3].set_title("Slot 5 Mask")
    axs[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/plots/ood_failure_analysis.png')
    print("OOD Failure analysis saved to results/plots/ood_failure_analysis.png")

if __name__ == '__main__':
    visualize_ood_failure()
