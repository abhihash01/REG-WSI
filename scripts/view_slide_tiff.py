from tiffslide import TiffSlide
import matplotlib.pyplot as plt
import os
import glob

# Set your directories
input_dir = "../competition_data/slide_folder"
out_dir = "../running_dir/competition_data/slide_images"
os.makedirs(out_dir, exist_ok=True)

# Get all TIFF files (change '*.tiff' to '*.tif' if needed)
tiff_files = sorted(glob.glob(os.path.join(input_dir, "*.tiff")))

for slide_path in tiff_files:
    try:
        slide = TiffSlide(slide_path)
        level = slide.get_best_level_for_downsample(32)
        dims = slide.level_dimensions[level]
        thumbnail = slide.read_region((0, 0), level, dims).convert("RGB")

        filename = os.path.splitext(os.path.basename(slide_path))[0]
        out_path = os.path.join(out_dir, f"{filename}.png")

        plt.imshow(thumbnail)
        plt.axis("off")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()  # Prevents memory issues in loops

        print(f"Processed and saved: {out_path}")
    except Exception as e:
        print(f"Failed to process {slide_path}: {e}")
