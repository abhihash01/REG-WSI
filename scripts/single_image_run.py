import torch
import h5py
from transformers import BioGptTokenizer
from histogpt.helpers.patching import main, PatchingConfigs
from transformers import BioGptConfig
from histogpt.models import HistoGPTForCausalLM, PerceiverResamplerConfig
from histogpt.helpers.inference import generate
import matplotlib.pyplot as plt
import os
from openslide import OpenSlide




device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

histogpt = HistoGPTForCausalLM(BioGptConfig(), PerceiverResamplerConfig())
histogpt = histogpt.to(device)


#loading the HistoGPT model
PATH = '../running_dir/histogpt-1b-6k-pruned.pth?download=true'
state_dict = torch.load(PATH, map_location=device)
histogpt.load_state_dict(state_dict, strict=True)


tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

prompt = 'Final diagnosis:'
prompt = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)


#patching and extracting features

configs = PatchingConfigs()
configs.slide_path = '../running_dir/slide_folder'
configs.save_path = '../running_dir/save_folder'
configs.model_path = '../running_dir/ctranspath.pth?download=true'
configs.patch_size = 256
configs.white_thresh = [170, 185, 175]
configs.edge_threshold = 2
configs.resolution_in_mpp = 0.0
configs.downscaling_factor = 4.0
configs.batch_size = 16

main(configs)



#load the image features

with h5py.File('../running_dir/save_folder/h5_files/256px_ctranspath_0.0mpp_4.0xdown_normal/2023-03-06 23.51.44.h5', 'r') as f:
    features = f['feats'][:]
    features = torch.tensor(features).unsqueeze(0).to(device)



#create textural features from prompt

tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

prompt = 'Final diagnosis:'
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


#printing output

#reading original image
slide_path = '../running_dir/slide_folder/2023-03-06 23.51.44.ndpi'
slide = OpenSlide(slide_path)

level = slide.get_best_level_for_downsample(32)
downsampled_dimensions = slide.level_dimensions[level]

thumbnail = slide.read_region((0,0), level, downsampled_dimensions)
thumbnail = thumbnail.convert("RGB")

output_dir = "../outputs/slide-output"
os.makedirs(output_dir, exist_ok=True)  #

plt.imshow(thumbnail)
plt.axis('off')
plt.savefig(os.path.join(output_dir, 'output.png'), bbox_inches='tight')


with open(os.path.join(output_dir, "decoded.txt"), "w") as f:
    #f.write(textwrap.fill(decoded, width=64))
    f.write(decoded)



