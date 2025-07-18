import os
import glob
import json
import torch
import h5py
from transformers import BioGptTokenizer, BioGptConfig
from histogpt.models import HistoGPTForCausalLM, PerceiverResamplerConfig
from histogpt.helpers.inference import generate

# Set device
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# Model and tokenizer setup
histogpt = HistoGPTForCausalLM(BioGptConfig(), PerceiverResamplerConfig()).to(device)
state_dict = torch.load('../running_dir/histogpt-1b-6k-pruned.pth?download=true', map_location=device)
histogpt.load_state_dict(state_dict, strict=True)
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

# Directory with h5 feature files
features_dir = '../competition_data/save_folder/h5_files/256px_ctranspath_0.0mpp_4.0xdown_normal/'
h5_files = sorted(glob.glob(os.path.join(features_dir, "*.h5")))

# PROMPT
prompt_text = 'Mention Organ and Cancer type'
prompt_tensor = torch.tensor(tokenizer.encode(prompt_text)).unsqueeze(0).to(device)

bulk_results = []

for path in h5_files:
    try:
        # Load features
        with h5py.File(path, 'r') as f:
            features = f['feats'][:]
        features = torch.tensor(features).unsqueeze(0).to(device)
        
        # Generate output (assume single image in file)
        with torch.no_grad():
            output = generate(
                model=histogpt,
                prompt=prompt_tensor,
                image=features,
                length=256,
                top_k=40,
                top_p=0.95,
                temp=0.7,
                device=device
            )
        decoded = tokenizer.decode(output[0, 1:])
        bulk_results.append({
            "id": os.path.basename(path),
            "report": decoded.strip()
        })
    except Exception as e:
        print(f'Error processing {path}: {e}')

# Save output in JSON format
output_dir = "../running_dir/competition_data/"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "batch_reports.json"), "w") as f:
    json.dump(bulk_results, f, indent=2)
