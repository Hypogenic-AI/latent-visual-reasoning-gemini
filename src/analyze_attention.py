
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.modules import SlotVideoModel
import seaborn as sns

def analyze_attention():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Collision Video
    video = torch.load('datasets/collision/collision.pt').to(device)
    
    model = SlotVideoModel(num_slots=6, slot_dim=64).to(device)
    model.load_state_dict(torch.load('results/models/slot_model.pth'))
    model.eval()
    
    input_seq = video[:, :-1]
    
    with torch.no_grad():
        out = model(input_seq)
        
    # out['attn_weights']: (B, T, N, N)
    # This is the attention matrix: [batch, time, target_slot, source_slot]
    # It tells us: For updating Target Slot, how much did it attend to Source Slot?
    
    attn = out['attn_weights'][0] # (T, N, N)
    T = attn.shape[0]
    
    # Identify Red and Blue slots again
    masks = out['masks'][0] # (T, N, 1, H, W)
    img_0 = input_seq[0, 0]
    
    # Find Red Slot
    red_score = []
    blue_score = []
    
    # Red is channel 0, Blue is channel 2
    for n in range(6):
        mask = masks[0, n, 0]
        r_s = (mask * img_0[0]).sum().item()
        b_s = (mask * img_0[2]).sum().item()
        red_score.append(r_s)
        blue_score.append(b_s)
        
    red_idx = np.argmax(red_score)
    blue_idx = np.argmax(blue_score)
    
    print(f"Red Slot: {red_idx}, Blue Slot: {blue_idx}")
    
    # We want to see: Does Red Slot attend to Blue Slot more during collision?
    # Collision happens around middle of video (20 frames).
    # Balls start at 20 and 44. Vel +/- 2. Dist 24.
    # Approach speed 4. Time to impact = 24 / 4 = 6 frames (radius 4 each -> contact at dist 8).
    # Actually center-to-center dist starts at 24. Contact when dist <= 8.
    # Travel dist = 24 - 8 = 16.
    # Time = 16 / 4 = 4 frames.
    # So collision around t=4 to t=6?
    
    # Let's plot Attention(Target=Red, Source=Blue) over time
    
    red_attend_blue = attn[:, red_idx, blue_idx].cpu().numpy()
    blue_attend_red = attn[:, blue_idx, red_idx].cpu().numpy()
    
    # Self attention
    red_self = attn[:, red_idx, red_idx].cpu().numpy()
    blue_self = attn[:, blue_idx, blue_idx].cpu().numpy()
    
    time_steps = np.arange(T)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, red_attend_blue, '-o', label='Red attends Blue', color='purple')
    plt.plot(time_steps, blue_attend_red, '-x', label='Blue attends Red', color='orange')
    plt.plot(time_steps, red_self, '--', label='Red Self-Attn', color='red', alpha=0.5)
    plt.plot(time_steps, blue_self, '--', label='Blue Self-Attn', color='blue', alpha=0.5)
    
    plt.xlabel("Time Step")
    plt.ylabel("Attention Weight")
    plt.title("Transformer Attention Dynamics During Collision")
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/attention_dynamics.png')
    print("Attention plot saved to results/plots/attention_dynamics.png")
    
    # Also save the full heatmap for the peak collision frame
    # Find max attention frame
    peak_frame = np.argmax(red_attend_blue)
    print(f"Peak Attention at Frame: {peak_frame}")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(attn[peak_frame].cpu().numpy(), annot=True, fmt=".2f", cmap='viridis')
    plt.title(f"Attention Matrix at Peak Collision (t={peak_frame})")
    plt.xlabel("Source Slot")
    plt.ylabel("Target Slot")
    plt.savefig('results/plots/peak_attention_matrix.png')
    print("Peak Attention Matrix saved.")

if __name__ == '__main__':
    analyze_attention()
