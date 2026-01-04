
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.modules import SlotVideoModel
from tqdm import tqdm

def analyze_causality():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    test_data = torch.load('datasets/bouncing_balls/test.pt')
    
    model = SlotVideoModel(num_slots=6, slot_dim=64).to(device)
    model.load_state_dict(torch.load('results/models/slot_model.pth'))
    model.eval()
    
    # We focus on the Predictor module: model.predictor
    # input: (B, N, D) -> output: (B, N, D)
    # We want to measure dy_j / dx_i
    
    interaction_matrices = []
    
    # Use first 20 videos
    videos = test_data[:20].to(device)
    
    print("Computing Interaction Matrices...")
    with torch.no_grad():
        # First, we need to get the slots for these videos
        # We run the encoder + slot attention
        B, T, C, H, W = videos.shape
        flat_video = videos.reshape(B * T, C, H, W)
        features = model.encoder(flat_video)
        slots = model.slot_attention(features) # (B*T, N, D)
        slots = slots.reshape(B, T, -1, 64)
        
        # Now iterate over time steps to test the predictor
        # Predictor takes slots at t and predicts t+1
        # We perform perturbation analysis on the predictor
        
        predictor = model.predictor
        
        for b in range(B):
            for t in range(T-1):
                current_slots = slots[b, t].unsqueeze(0) # (1, N, D)
                
                # Baseline prediction
                # The predictor in modules.py does:
                # interacted = transformer(current_slots)
                # delta = linear(interacted)
                # next = current + delta
                
                # We want to see how much 'delta' for slot j changes when we perturb slot i
                
                base_delta = predictor.predictor(predictor.transformer(current_slots)) # (1, N, D)
                
                matrix = np.zeros((6, 6))
                
                for i in range(6):
                    # Perturb slot i
                    perturbed_slots = current_slots.clone()
                    noise = torch.randn_like(perturbed_slots[:, i]) * 0.1 # Small perturbation
                    perturbed_slots[:, i] += noise
                    
                    # New prediction
                    new_delta = predictor.predictor(predictor.transformer(perturbed_slots))
                    
                    # Measure change in all slots
                    diff = (new_delta - base_delta).abs().mean(dim=-1).squeeze(0) # (N,)
                    
                    # Normalize by the magnitude of perturbation to get sensitivity
                    # But noise magnitude is constant-ish. Let's just use raw diff.
                    
                    matrix[i, :] = diff.cpu().numpy()
                    
                interaction_matrices.append(matrix)
                
    # Average Interaction Matrix
    avg_matrix = np.mean(interaction_matrices, axis=0)
    
    # Normalize for visualization (so diagonal is 1.0? No, let's keep relative scales)
    # Diagonal elements (Self-influence) should be high.
    # Off-diagonal elements (Interaction) represent physics/collisions.
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_matrix, annot=True, fmt=".4f", cmap="viridis")
    plt.title("Average Slot Interaction Matrix\n(Sensitivity of Slot j to Perturbation in Slot i)")
    plt.xlabel("Affected Slot (j)")
    plt.ylabel("Perturbed Slot (i)")
    plt.savefig('results/plots/interaction_matrix.png')
    print("Interaction matrix saved to results/plots/interaction_matrix.png")
    
    # Calculate Interaction Score
    # Ratio of Off-Diagonal to Diagonal
    diag_mean = np.mean(np.diag(avg_matrix))
    off_diag_mean = np.mean(avg_matrix[~np.eye(6, dtype=bool)])
    
    interaction_score = off_diag_mean / diag_mean
    print(f"Diagonal Mean (Self-Consistency): {diag_mean:.4f}")
    print(f"Off-Diagonal Mean (Interaction): {off_diag_mean:.4f}")
    print(f"Interaction Score (Off/Diag): {interaction_score:.4f}")
    
    with open('results/causality_metrics.txt', 'w') as f:
        f.write(f"Diagonal Mean: {diag_mean:.4f}\n")
        f.write(f"Off-Diagonal Mean: {off_diag_mean:.4f}\n")
        f.write(f"Interaction Score: {interaction_score:.4f}\n")

if __name__ == '__main__':
    analyze_causality()
