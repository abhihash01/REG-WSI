# install once
# pip install tiffslide matplotlib

from tiffslide import TiffSlide          # only line that changes
import matplotlib.pyplot as plt
import os

slide_path = "../running_dir/competition_data/slide_folder/PIT_01_00002_01.tiff"
slide      = TiffSlide(slide_path)       # works for ordinary or pyramidal TIFF

# pick a downsample similar to OpenSlide
level      = slide.get_best_level_for_downsample(32)
dims       = slide.level_dimensions[level]

thumbnail  = slide.read_region((0, 0), level, dims).convert("RGB")

out_dir = "../outputs/slide-image"
os.makedirs(out_dir, exist_ok=True)

plt.imshow(thumbnail)
plt.axis("off")
plt.savefig(os.path.join(out_dir, "output.png"), bbox_inches="tight")
