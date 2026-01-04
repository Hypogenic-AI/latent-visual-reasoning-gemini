import json
import os
from datasets import load_dataset

os.makedirs("datasets/clevr_sample", exist_ok=True)

try:
    print("Attempting to stream CLEVR sample...")
    # Try HuggingFaceM4/clevr with trust_remote_code=True
    ds = load_dataset("HuggingFaceM4/clevr", split="validation", streaming=True, trust_remote_code=True)
    
    samples = []
    for i, item in enumerate(ds):
        if i >= 10: break
        # Save image to disk
        img_path = f"datasets/clevr_sample/image_{i}.png"
        item['image'].save(img_path)
        item['image'] = img_path # Replace object with path for JSON
        samples.append(item)
    
    with open("datasets/clevr_sample/samples.json", "w") as f:
        json.dump(samples, f, indent=2)
    
    print("Successfully saved 10 CLEVR samples.")

except Exception as e:
    print(f"Failed to load CLEVR: {e}")
