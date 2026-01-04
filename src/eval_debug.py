
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.modules import SlotVideoModel
from src.baseline import ConvPredictor
import os

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    test_data = torch.load('datasets/bouncing_balls/test.pt')
    
    # Init Models
    slot_model = SlotVideoModel(num_slots=6, slot_dim=64).to(device)
    baseline_model = ConvPredictor(hidden_dim=32).to(device)
    
    # Load Weights
    if os.path.exists('results/models/slot_model.pth'):
        slot_model.load_state_dict(torch.load('results/models/slot_model.pth'))
        baseline_model.load_state_dict(torch.load('results/models/baseline_model.pth'))
        print("Models loaded.")
    else:
        print("Models not found, running with random weights.")
        
    slot_model.eval()
    baseline_model.eval()
    
    with torch.no_grad():
        test_video = test_data[:2].to(device) # Take 2 examples
        # input_seq = test_video[:, :-1] 
        # Reducing sequence length to see if it fixes "index 18 out of 9"
        # If it was 10 during training, maybe model has some state? No, it's stateless.
        
        input_seq = test_video[:, :-1]
        target_seq = test_video[:, 1:]
        
        print(f"Test input shape: {input_seq.shape}")
        
        out_slot = slot_model(input_seq)
        out_base = baseline_model(input_seq)
        
        print(f"Slot output shape: {out_slot['pred_next_img'].shape}")
        
        t_idx = input_seq.shape[1] - 1
        print(f"t_idx: {t_idx}")
        
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        
        out = out_slot
        
        for i in range(2):
            # Input
            axs[i, 0].imshow(input_seq[i, t_idx].permute(1, 2, 0).cpu().numpy())
            axs[i, 0].set_title(f"Input t={t_idx}")
            axs[i, 0].axis('off')
            
            # Target
            axs[i, 1].imshow(target_seq[i, t_idx].permute(1, 2, 0).cpu().numpy())
            axs[i, 1].set_title(f"Target t={t_idx+1}")
            axs[i, 1].axis('off')
            
            # Baseline Pred
            axs[i, 2].imshow(out_base[i, t_idx].permute(1, 2, 0).cpu().numpy())
            axs[i, 2].set_title("Baseline Pred")
            axs[i, 2].axis('off')
            
            # Slot Pred
            axs[i, 3].imshow(out['pred_next_img'][i, t_idx].permute(1, 2, 0).cpu().numpy())
            axs[i, 3].set_title("Slot Pred")
            axs[i, 3].axis('off')
            
            # Slot Masks
            masks = out['masks'][i, t_idx]
            mask_viz = torch.zeros(3, 64, 64).to(device)
            colors = torch.tensor([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1], [1,0,1]]).to(device)
            
            for k in range(min(masks.shape[0], 6)):
                m = masks[k]
                c = colors[k].view(3, 1, 1)
                mask_viz += m * c
                
            axs[i, 4].imshow(mask_viz.permute(1, 2, 0).cpu().numpy())
            axs[i, 4].set_title("Slot Attention")
            axs[i, 4].axis('off')
            
        plt.tight_layout()
        plt.savefig('results/plots/qualitative_comparison_debug.png')
        print("Plots saved.")

if __name__ == '__main__':
    evaluate()
