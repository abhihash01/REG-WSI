# takes encoded file and generates the textual output
import torch
import h5py
from transformers import BioGptTokenizer
from transformers import BioGptConfig
from histogpt.models import HistoGPTForCausalLM, PerceiverResamplerConfig
from histogpt.helpers.inference import generate
import matplotlib.pyplot as plt
import os



device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

histogpt = HistoGPTForCausalLM(BioGptConfig(), PerceiverResamplerConfig())
histogpt = histogpt.to(device)


PATH = '../running_dir/histogpt-1b-6k-pruned.pth?download=true'
state_dict = torch.load(PATH, map_location=device)
histogpt.load_state_dict(state_dict, strict=True)




#features_file = '../running_dir/save_folder/h5_files/256px_ctranspath_0.0mpp_4.0xdown_normal/2023-03-06 23.51.44.h5'
features_file = '../running_dir/competition_data/save_folder/h5_files/256px_ctranspath_0.0mpp_4.0xdown_normal/PIT_01_00002_01.h5'
with h5py.File(features_file, 'r') as f:
    features = f['feats'][:]
    features = torch.tensor(features).unsqueeze(0).to(device)




#create textural features from prompt

tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

prompt = 'Give json with organ and cancer type like { organ : organ, cancer_type:cancer_type}'
prompt = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)


# generate text autoregressively


output = generate(
    model=histogpt,
    prompt=prompt,
    image=features,
    length=256,
    top_k=40,
    top_p=0.95,
    temp=0.7,
    device=device
)

decoded = tokenizer.decode(output[0, 1:])

output_dir = "../outputs/feature-output-tiff/"
os.makedirs(output_dir, exist_ok=True)


with open(os.path.join(output_dir, "decoded.txt"), "w") as f:
    #f.write(textwrap.fill(decoded, width=64))
    f.write(decoded)
