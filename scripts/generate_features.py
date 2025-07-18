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
configs.slide_path = '../running_dir/competition_data/slide_folder'
configs.save_path = '../running_dir/competition_data/save_folder'
configs.model_path = '../running_dir/ctranspath.pth?download=true'
configs.file_extension = '.tiff'
configs.patch_size = 256
configs.white_thresh = [170, 185, 175]
configs.edge_threshold = 2
configs.resolution_in_mpp = 0.0
configs.downscaling_factor = 4.0
configs.batch_size = 16

main(configs)









