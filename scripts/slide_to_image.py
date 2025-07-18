from openslide import OpenSlide
import os
#file to read the slides

slide_path = '/running_dir/competition_data/slide_folder/PIT_01_00002_01.tiff'
slide = OpenSlide(slide_path)

level = slide.get_best_level_for_downsample(32)
downsampled_dimensions = slide.level_dimensions[level]

thumbnail = slide.read_region((0,0), level, downsampled_dimensions)
thumbnail = thumbnail.convert("RGB")

output_dir = "../outputs/slide-image"
os.makedirs(output_dir, exist_ok=True)  #

plt.imshow(thumbnail)
plt.axis('off')
plt.savefig(os.path.join(output_dir, 'output.png'), bbox_inches='tight')