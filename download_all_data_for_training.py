"""Download OMERO images and ground-truth rectangles for training.

Edit the settings below, then run: python download_all_data_for_training.py
Requires the project's OMERO/CCIPy dependencies, NumPy, scikit-image and Pillow.
Each image CSV has columns class, x, y, width, height in output-image pixels.
Boxes are clipped to the 1024 x 1024 output; boxes outside it are omitted.
Images with ROIs but no matching ground-truth rectangles get a header-only CSV.
"""

# Settings: supply an OMERO session token and the group containing the datasets.
TARGET_DIRECTORY = "training_data"
OMERO_TOKEN = ""
DATASET_IDS = []  # Example: [1159, 1214]
OMERO_GROUP = "Emma-Josefsson-Lab"
OMERO_HOST = "omero-cci-cli.gu.se"
OMERO_PORT = "4064"

CLASS_COLORS = {"granule": (255, 255, 0), "cell": (0, 181, 255)}
SCALE = 4
OUTPUT_SIZE = 1024

import csv
import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.transform import resize
from omero.rtypes import unwrap
from ccipy.omero.cci_omero_connection import OmeroConnection
from ccipy.omero.omero_getter_ctx import OmeroGetterCtx
from ccipy.utils.cci_logger import CCILogger


def ground_truth_boxes(rois, image_id, offset_x, offset_y):
    """Read unnamed ROIs; identify classes by stroke RGB, ignoring alpha."""
    colors = {rgb: name for name, rgb in CLASS_COLORS.items()}
    rows = []
    for roi in rois:
        if unwrap(roi.getName()):
            continue
        for shape in roi.copyShapes():
            color = unwrap(shape.getStrokeColor())
            rgb = tuple((color >> shift) & 255 for shift in (24, 16, 8)) if color is not None else None
            name = colors.get(rgb)
            if name is None:
                continue
            try:
                x, y, width, height = (
                    float(unwrap(getattr(shape, method)()))
                    for method in ("getX", "getY", "getWidth", "getHeight")
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(f"Image {image_id}: {name} shape is not a valid rectangle") from exc
            if not all(math.isfinite(v) for v in (x, y, width, height)) or width <= 0 or height <= 0:
                raise ValueError(f"Image {image_id}: invalid {name} rectangle")
            x1 = max(0.0, x / SCALE + offset_x)
            y1 = max(0.0, y / SCALE + offset_y)
            x2 = min(float(OUTPUT_SIZE), (x + width) / SCALE + offset_x)
            y2 = min(float(OUTPUT_SIZE), (y + height) / SCALE + offset_y)
            if x2 > x1 and y2 > y1:
                rows.append((name, x1, y1, x2 - x1, y2 - y1))
    return rows


def downscaled_image(image):
    """Apply the same contrast stretch as the project's training pipeline."""
    width, height = image.getSizeX(), image.getSizeY()
    if (image.getSizeZ(), image.getSizeC(), image.getSizeT()) != (1, 1, 1):
        raise ValueError("Image must be single-plane grayscale")
    plane = image.getPrimaryPixels().getPlane(theZ=0, theC=0, theT=0)
    if plane.ndim != 2 or plane.shape != (height, width):
        raise ValueError("OMERO returned an unexpected image plane shape")
    pixels = plane[plane > 1000] if np.count_nonzero(plane < 1000) / plane.size > 0.1 else plane
    if pixels.size == 0:
        stretched = np.zeros(plane.shape, dtype=np.float32)
    else:
        lo, hi = np.percentile(pixels, (1, 99))
        stretched = (plane.astype(np.float32) - lo) / (hi - lo) if hi > lo else np.zeros(plane.shape, dtype=np.float32)
    # Pad the source at its bottom/right to preserve the exact 4:1 coordinate scale.
    pad_y, pad_x = (-height) % SCALE, (-width) % SCALE
    if pad_y or pad_x:
        stretched = np.pad(stretched, ((0, pad_y), (0, pad_x)))
    scaled = resize(stretched, (stretched.shape[0] // SCALE, stretched.shape[1] // SCALE), anti_aliasing=True)
    scaled = np.clip(scaled * 255, 0, 255).astype(np.uint8)
    scaled_height, scaled_width = scaled.shape
    offset_x = (OUTPUT_SIZE - scaled_width) // 2 if scaled_width <= OUTPUT_SIZE else -((scaled_width - OUTPUT_SIZE) // 2)
    offset_y = (OUTPUT_SIZE - scaled_height) // 2 if scaled_height <= OUTPUT_SIZE else -((scaled_height - OUTPUT_SIZE) // 2)
    result = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8)
    source_x, source_y = max(0, -offset_x), max(0, -offset_y)
    dest_x, dest_y = max(0, offset_x), max(0, offset_y)
    copy_width = min(scaled_width - source_x, OUTPUT_SIZE - dest_x)
    copy_height = min(scaled_height - source_y, OUTPUT_SIZE - dest_y)
    result[dest_y:dest_y + copy_height, dest_x:dest_x + copy_width] = scaled[source_y:source_y + copy_height, source_x:source_x + copy_width]
    return Image.fromarray(result), offset_x, offset_y


def main():
    if not OMERO_TOKEN or not OMERO_GROUP or not DATASET_IDS:
        raise ValueError("Set OMERO_TOKEN, OMERO_GROUP and DATASET_IDS at the top of the script")
    if any(type(i) is not int or i <= 0 for i in DATASET_IDS) or len(set(DATASET_IDS)) != len(DATASET_IDS):
        raise ValueError("DATASET_IDS must contain unique positive integers")
    output = Path(TARGET_DIRECTORY).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("omero_training_download")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    CCILogger._logger = logger
    connection = OmeroConnection(OMERO_HOST, OMERO_PORT, OMERO_TOKEN)
    try:
        connection.set_group_name_for_session(OMERO_GROUP)
        with OmeroGetterCtx(connection) as getter, (output / "manifest.csv").open("w", newline="", encoding="utf-8") as manifest_file:
            manifest = csv.writer(manifest_file)
            manifest.writerow(("dataset_id", "image_id"))
            manifest_file.flush()
            for dataset_id in DATASET_IDS:
                if connection.get_dataset(dataset_id) is None:
                    raise ValueError(f"Dataset {dataset_id} is unavailable")
                for image_id in getter.get_image_ids_from_dataset(dataset_id):
                    rois = list(getter.get_rois_for_image(image_id))
                    if not rois:
                        print(f"Skipping image {image_id}: no ROIs")
                        continue
                    image = connection.get_image(image_id)
                    if image is None:
                        raise ValueError(f"Image {image_id} is unavailable")
                    stem = f"{dataset_id}_{image_id}"
                    processed, offset_x, offset_y = downscaled_image(image)
                    boxes = ground_truth_boxes(rois, image_id, offset_x, offset_y)
                    processed.save(output / f"{stem}.png")
                    with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow(("class", "x", "y", "width", "height"))
                        writer.writerows(boxes)
                    manifest.writerow((dataset_id, image_id))
                    manifest_file.flush()
                    print(f"Downloaded {stem}: {len(boxes)} ground-truth boxes")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
