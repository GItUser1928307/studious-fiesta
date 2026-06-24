#!/usr/bin/env python3
"""
Save AI Model Locally for Publishing
Downloads and saves model files to local directory
"""
import os
import torch
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SAVING AI MODEL FOR PUBLICATION")
print("="*60)

# Create directory
save_dir = "AI_Model_Package"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(f"{save_dir}/model_files", exist_ok=True)

print(f"\nSaving to: {save_dir}/")

# Load model from HuggingFace
print("\nLoading model from HuggingFace...")
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "roneneldan/TinyStories-33M"
tokenizer_name = "EleutherAI/gpt-neo-125M"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save locally
print("\nSaving model files...")
model.save_pretrained(f"{save_dir}/model_files")
tokenizer.save_pretrained(f"{save_dir}/model_files")

# Get size
import glob
total_size = sum(os.path.getsize(f) for f in glob.glob(f"{save_dir}/model_files/**/*", recursive=True) if os.path.isfile(f))

print(f"\n✅ Model saved!")
print(f"   Location: {save_dir}/model_files/")
print(f"   Size: {total_size/1e6:.1f} MB")

# Create README
readme = """# AI Language Model - 68M Parameters

## Model Details
- **Name**: TinyStories-33M
- **Parameters**: 68,514,048 (68.5M)
- **Architecture**: GPT-Neo (Transformer Decoder)
- **Source**: HuggingFace (roneneldan/TinyStories-33M)

## Hardware Requirements
- CPU: Intel i3 or equivalent
- RAM: 2GB minimum (4GB recommended)
- Storage: ~300MB

## How to Use

### Python
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./model_files")
model = AutoModelForCausalLM.from_pretrained("./model_files")

prompt = "Once upon a time"
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### CLI
```bash
python run_model.py
```

## License
MIT

## Source
https://huggingface.co/roneneldan/TinyStories-33M
"""

with open(f"{save_dir}/README.md", "w") as f:
    f.write(readme)

# Create main run script
run_script = '''#!/usr/bin/env python3
"""AI Model Runner"""
import os, torch, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading model from local files...")
tokenizer = AutoTokenizer.from_pretrained("model_files")
model = AutoModelForCausalLM.from_pretrained("model_files")
model.eval()

print("Model ready! Type your prompts.")

while True:
    prompt = input("\\nYou: ")
    if prompt.lower() == "quit": break
    
    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=256)
    outputs = model.generate(inputs, max_new_tokens=80, temperature=0.7, top_p=0.92, do_sample=True)
    print("AI:", tokenizer.decode(outputs[0], skip_special_tokens=True))
'''

with open(f"{save_dir}/run_model.py", "w") as f:
    f.write(run_script)

print("\n" + "="*60)
print("PUBLICATION PACKAGE READY!")
print("="*60)
print(f"""
Package location: {save_dir}/

Contents:
  - model_files/  (the AI model)
  - README.md    (documentation)
  - run_model.py (launcher)

To publish: Zip the entire folder and share!
""")