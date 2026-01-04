
import numpy as np
import torch
import os

def generate_collision_video(size=64, num_frames=20):
    """
    Generates a single video with 2 balls colliding head-on.
    """
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)
    
    # Ball 1: Left, moving Right
    b1 = {
        'pos': np.array([20.0, 32.0]),
        'vel': np.array([2.0, 0.0]),
        'radius': 4.0,
        'color': np.array([1.0, 0.0, 0.0]) # Red
    }
    
    # Ball 2: Right, moving Left
    b2 = {
        'pos': np.array([44.0, 32.0]),
        'vel': np.array([-2.0, 0.0]),
        'radius': 4.0,
        'color': np.array([0.0, 0.0, 1.0]) # Blue
    }
    
    balls = [b1, b2]
    frames = []
    
    for t in range(num_frames):
        frame = np.zeros((size, size, 3))
        
        # Physics (Simplified)
        for i, ball in enumerate(balls):
            ball['pos'] += ball['vel']
            
        # Collision Check
        dist = np.linalg.norm(balls[0]['pos'] - balls[1]['pos'])
        if dist < (balls[0]['radius'] + balls[1]['radius']):
            # Elastic collision (1D simple swap for same mass)
            v1 = balls[0]['vel'].copy()
            v2 = balls[1]['vel'].copy()
            balls[0]['vel'] = v2
            balls[1]['vel'] = v1
            
        # Draw
        for ball in balls:
            d = np.sqrt((xx - ball['pos'][0])**2 + (yy - ball['pos'][1])**2)
            mask = d <= ball['radius']
            frame[mask] = ball['color']
            
        frames.append(frame)
        
    video = np.array([frames]) # (1, T, H, W, C)
    video = video.transpose(0, 1, 4, 2, 3) # (1, T, C, H, W)
    return torch.tensor(video, dtype=torch.float32)

if __name__ == '__main__':
    video = generate_collision_video()
    os.makedirs('datasets/collision', exist_ok=True)
    torch.save(video, 'datasets/collision/collision.pt')
    print("Collision video saved.")
