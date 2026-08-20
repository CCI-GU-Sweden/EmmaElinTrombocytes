import argparse
from pathlib import Path
from datetime import datetime
import os
import yaml
from yaml import Loader
import csv
import distutils.util
from ccipy.utils.cci_logger import CCILogger
from ccipy.omero.cci_omero_connection import OmeroConnection
from ccipy.omero.omero_getter_ctx import OmeroGetterCtx
from ccipy.omero.omero_colors import omero_rint_to_rgba
from ccipy.utils.roi_geometry import RoiGeometry
from omero.rtypes import unwrap
from skimage.transform import resize
from skimage.filters import threshold_otsu, median, gaussian
from skimage.morphology import remove_small_objects, disk, remove_small_holes, closing
from skimage.measure import label, regionprops
from skimage.transform import warp_polar
from PIL import Image
import tifffile as tiff
import numpy as np
import config

label_names = [(0,"Cell"), (1,"Granule"), (2,"Atypical Granule"), (3,"Unclear if Granule")]

def get_class_name_from_id(class_id: int) -> str:
    for cid, name in label_names:
        if cid == class_id:
            return name
    CCILogger.warning(f"Class id {class_id} not found in label names.")
    return "Unknown"

def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--config_name',
        type=str,
        required=False,
        help='Name of the configuration to use from the config.yaml file') 
    
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Token for OMERO connection"
    )
    
    parser.add_argument(
        "--group",
        type=str,
        required=False,
        help="Group to use for the OMERO session"
    )
    
    parser.add_argument(
        "--use_test_host",
        type=distutils.util.strtobool,
        required=False,
        default=False,
        help="Use test host"
    )

                        
def read_config_from_file(config_type: str,  config_name: str = "", config_file: str = "config.yaml") -> tuple[dict, dict] | None:
    with open(config_file, 'r') as f:
        config_dict = yaml.load(f, Loader=Loader)
        configs = config_dict.get(config_type, None)
        if configs is None:
            CCILogger.error(f"Config type {config_type} not found in config file {config_file}")
            return None
        common = configs.get("common", {})
        conf_list = configs.get("configs", [])
        conf = next(item for item in conf_list if item["name"] == config_name)
        if conf is None:
            CCILogger.error(f"Config name {config_name} not found in config file {config_file}")
            return None
        
        return common, conf

def init(session_token: str, session_group: str, use_test_host: bool = False, init_log: bool = True) -> OmeroConnection:

    if init_log:
        CCILogger.setup_logger("logfile.log", "omero_test")

    if use_test_host:
        connection = OmeroConnection(config.OMERO_TEST_HOST,config.OMERO_PORT,session_token)
    else:
        connection = OmeroConnection(config.OMERO_HOST,config.OMERO_PORT,session_token)
    
    connection.set_group_name_for_session(session_group)
    return connection

def color_stretch(img, low_cut_off, low_pt=1, high_pt=99):

    #check if large portion are zeroes
    
    dark = np.extract(img < low_cut_off, img)
    if (frac := len(dark) / img.size) > config.FRACTION_DARK_TO_USE_CUTOFF:
        CCILogger.info(f"Image has more ({frac*100:.2f}%) pixels than {config.FRACTION_DARK_TO_USE_CUTOFF*100}% below low_cutoff={low_cut_off}, filtering")

        img_bool = img > low_cut_off
        img_filterd = img[img_bool]
        lo = np.percentile(img_filterd, low_pt)
        hi = np.percentile(img_filterd, high_pt)
    else:
        lo = np.percentile(img, low_pt)
        hi = np.percentile(img, high_pt)
    
    img = img.astype(np.float32)

    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)

    stretched = (img - lo) / (hi - lo)
    return stretched


def downscale_data(image: Path, target_size: tuple[int, int]) -> Path | None:
    if image.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        return None
    img = tiff.imread(image)
    original_size = img.shape[:2]
    if original_size != target_size:
        CCILogger.info(f"Color stretching image {image.name} of type {img.dtype}")
        stretched = color_stretch(img, config.LOW_CUTOFF)
        
        CCILogger.info(f"Downscaling image {image.name} from {original_size} to {target_size}")
        if len(img.shape) == 3:
            downscaled_img = resize(stretched, target_size + (img.shape[2],), anti_aliasing=True)
        else:
            downscaled_img = resize(stretched, target_size, anti_aliasing=True)
            
        downscaled_img = downscaled_img * 255.0

        downscaled_img = np.clip(downscaled_img, 0, 255).astype(np.uint8)
        #downscaled_img = (downscaled_img * 65535).astype(np.uint16)
        im = Image.fromarray(downscaled_img)
        if str(image).endswith("ome.tiff"):
            img_name_png = str(image).replace(".ome.tiff",".png")
        if str(image).endswith("ome.tif"):
            img_name_png = str(image).replace(".ome.tif",".png")
    
        im.save(img_name_png)

        #tiff.imwrite(img_name_tif, downscaled_img)
        os.unlink(image)
        return Path(img_name_png)
    else:
        CCILogger.info(f"Image {image.name} already at target size {target_size}, skipping downscale.")
    return image

def download_downscaled_image(connection: OmeroConnection, img_id: int, images_dir: Path, img_size: int = 512) -> tuple[Path | None, int, int]:
    with OmeroGetterCtx(connection) as getter:
        img = connection.get_image(img_id)
        img_width = img.getSizeX()
        img_height = img.getSizeY()
        pixels = img.getPrimaryPixels()._obj
        px_x = pixels.getPhysicalSizeX()
        px_y = pixels.getPhysicalSizeY()

        getter.download_original_image_file(img_id, images_dir)
        new_img_name = downscale_data(images_dir / Path(img.getName()), (img_size, img_size))
        
        return new_img_name, img_width, img_height, px_x.getValue(), px_y.getValue(), px_x.getUnit()


def geometry_to_class_definitions(geometry: RoiGeometry, skiplist = []) -> int:
    # Blue (rgb(0, 181, 255)) = Cell
    # Yellow (rgb(255, 255, 0)) = Granule
    # Purple (rgb(152, 0, 255) = Atypical granule
    # Red (rgb(255, 0, 0)) = Unclear if granule
    color = geometry.get_color()
    r, g, b, a = omero_rint_to_rgba(color)
    if r == 0 and g == 181 and b == 255:
        if 0 in skiplist:
            CCILogger.info(f"{label_names[0][1]} found but we dont use it for training, skipping")
            raise ValueError(f"Skipping {label_names[0][1]} for training")
        return 0
    elif r == 255 and g == 255 and b == 0:
        if 1 in skiplist:
            CCILogger.info(f"{label_names[1][1]} found but we dont use it for training, skipping")
            raise ValueError(f"Skipping {label_names[1][1]} for training")
        return 1
    elif r == 152 and g == 0 and b == 255:
        if 2 in skiplist:
            CCILogger.info(f"{label_names[2][1]} found but we dont use it for training, skipping")
            raise ValueError(f"Skipping {label_names[2][1]} for training")
        return 2
    elif r == 255 and g == 0 and b == 0:
        if 3 in skiplist:
            CCILogger.info(f"{label_names[3][1]} found but we dont use it for training, skipping")
            raise ValueError(f"Skipping {label_names[3][1]} for training")
        return 3
    else:
        CCILogger.warning(f"Color on geometry is not according to spec {r} {g} {b}")
        raise ValueError("Wrong color")
    
    
def class_to_color(class_id: int) -> tuple[int, int, int]:
    if class_id == 0:
        return (0, 181, 255)
    elif class_id == 1:
        return (255, 255, 0)
    elif class_id == 2:
        return (152, 0, 255)
    elif class_id == 3:
        return (255, 0, 0)
    else:
        CCILogger.warning(f"Class id {class_id} not according to spec")
        raise ValueError("Wrong class id")

def color_to_class(color: tuple[int, int, int]) -> int:
    if color == (0, 181, 255):
        return 0
    elif color == (255, 255, 0):
        return 1
    elif color == (152, 0, 255):
        return 2
    elif color == (255, 0, 0):
        return 3
    else:
        CCILogger.warning(f"Color {color} not according to spec")
        raise ValueError("Wrong color")

def generate_date_directory():
    # Get current date and time
    now = datetime.now()

    # Format strings
    date_str = now.strftime("%Y-%m-%d")   # e.g., 2026-02-25
    time_str = now.strftime("%H-%M")      # e.g., 14-37

    # Create directory path
    base_path = Path(date_str)
    full_path = base_path / time_str
    return full_path

def _unwrap_omero_string(value):
    print(f"        Unwrapping value: {value} of type {type(value)}")
    if hasattr(value, "val"):
        print(f"        Value has 'val' attribute: {value.val}")
    if value is None:
        return None
    return value.val if hasattr(value, "val") else str(value)

def _unwrap_omero_id(value):
    if value is None:
        return None
    if hasattr(value, "getValue"):
        return value.getValue()
    if hasattr(value, "val"):
        return value.val
    return value

def crop_from_predicted_bbox(img_arr, rect):

    img_width = img_arr.shape[1]
    img_height = img_arr.shape[0]

    bbox_x = float(rect.x)
    bbox_y = float(rect.y)
    bbox_w = float(rect.width)
    bbox_h = float(rect.height)

    # 10% margin on each side
    margin_x = 0.05 * bbox_w
    margin_y = 0.05 * bbox_h

    crop_x0 = int(np.floor(bbox_x - margin_x))
    crop_y0 = int(np.floor(bbox_y - margin_y))
    crop_x1 = int(np.ceil(bbox_x + bbox_w + margin_x))
    crop_y1 = int(np.ceil(bbox_y + bbox_h + margin_y))

    # Clamp to image bounds
    crop_x0 = max(0, crop_x0)
    crop_y0 = max(0, crop_y0)
    crop_x1 = min(img_width, crop_x1)
    crop_y1 = min(img_height, crop_y1)

    crop_w = crop_x1 - crop_x0
    crop_h = crop_y1 - crop_y0

    crop_arr = img_arr[crop_y0:crop_y1, crop_x0:crop_x1]

    return crop_arr

def get_crops_and_rois_from_dataset(omero_conn, dataset_id, crops_dir, num_crops=None):
    current_user_id = _unwrap_omero_id(omero_conn.get_user_id())
    if current_user_id is None:
        CCILogger.warning("Could not resolve current OMERO user id. Skipping ROI removal for safety.")
        return 0
    
    csv_path = crops_dir / "crop_info.csv"
    crop_rows = []
    granule_counter = 0

    with OmeroGetterCtx(omero_conn) as getter:
        image_ids = list(getter.get_image_ids_from_dataset(dataset_id))
        print(f"Number of image IDs = {len(image_ids)}")
        us = omero_conn.get_update_service()

        # test_ids = [24429, 24427, 24382, 24430, 24381]
        if not num_crops:
            crop_bounds = -1
        else:
            crop_bounds = num_crops
        
        for image_id in image_ids[:crop_bounds]:
            # getter.download_original_image_file(image_id, images_dir)
            if image_id == 24428:
                continue
            print(f"\nIMAGE ID {image_id}")
            img = omero_conn.get_image(image_id)
            img_width = img.getSizeX()
            img_height = img.getSizeY()
            pixels = img.getPrimaryPixels()._obj
            px_x = pixels.getPhysicalSizeX()._value
            px_y = pixels.getPhysicalSizeY()._value
            print(f'Image width and height: {(img_width, img_height)}')
            print(f'Physical size per pixel: {(px_x, px_y)}')
            pixels = img.getPrimaryPixels()

            img_arr = pixels.getPlane(
                theZ=0,
                theC=0,
                theT=0
            )

            rois = getter.get_rois_for_image(image_id)
            for roi in rois:
                # roi_name = roi.getName()._val if hasattr(roi, "getName") else None
                # print(f'\nROI: {type(roi)}\n')
                # print(f'  ROI name: {roi_name}')
                for shape in roi.copyShapes():
                    shape_class = shape.getTextValue()._val if hasattr(shape, "getTextValue") else None
                    print(f'  Shape class: {shape_class}')
                    shape_x = shape.getX()._val if hasattr(shape, "getX") else None
                    shape_y = shape.getY()._val if hasattr(shape, "getY") else None
                    shape_width = shape.getWidth()._val if hasattr(shape, "getWidth") else None
                    shape_height = shape.getHeight()._val if hasattr(shape, "getHeight") else None
                    print(f'  Shape bbox: x={shape_x:.2f}, y={shape_y:.2f}, width={shape_width:.2f}, height={shape_height:.2f}\n')
                    shape_color = shape.getStrokeColor()._val if hasattr(shape, "getFillColor") else None
                    print(f'  Shape color: {shape_color}\n')

                    shape_class_str = shape_class or ""
                    if "(" in shape_class_str and ")" in shape_class_str:
                        parsed_class = shape_class_str.split("(", 1)[0].strip()
                        confidence = float(shape_class_str.split("(", 1)[1].split(")", 1)[0])
                    else:
                        parsed_class = shape_class_str.strip()
                        confidence = None
                    if parsed_class == "Granule":
                        granule_counter += 1

                        # Original bbox
                        bbox_x = float(shape_x)
                        bbox_y = float(shape_y)
                        bbox_w = float(shape_width)
                        bbox_h = float(shape_height)

                        # 10% margin on each side
                        margin_x = 0.05 * bbox_w
                        margin_y = 0.05 * bbox_h

                        crop_x0 = int(np.floor(bbox_x - margin_x))
                        crop_y0 = int(np.floor(bbox_y - margin_y))
                        crop_x1 = int(np.ceil(bbox_x + bbox_w + margin_x))
                        crop_y1 = int(np.ceil(bbox_y + bbox_h + margin_y))

                        # Clamp to image bounds
                        crop_x0 = max(0, crop_x0)
                        crop_y0 = max(0, crop_y0)
                        crop_x1 = min(img_width, crop_x1)
                        crop_y1 = min(img_height, crop_y1)

                        crop_w = crop_x1 - crop_x0
                        crop_h = crop_y1 - crop_y0

                        crop_arr = img_arr[crop_y0:crop_y1, crop_x0:crop_x1]

                        crop_filename = f"{image_id}_crop_g{granule_counter}.png"
                        crop_path = crops_dir / crop_filename

                        Image.fromarray(crop_arr).save(crop_path)

                        crop_rows.append({
                            "png": crop_filename,
                            "shape_class": parsed_class,
                            "confidence": confidence,
                            "bbox_x": bbox_x,
                            "bbox_y": bbox_y,
                            "bbox_w": bbox_w,
                            "bbox_h": bbox_h,
                            "crop_x": crop_x0,
                            "crop_y": crop_y0,
                            "crop_w": crop_w,
                            "crop_h": crop_h,
                        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "png",
                "shape_class",
                "confidence",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "crop_x",
                "crop_y",
                "crop_w",
                "crop_h",
            ],
        )
        writer.writeheader()
        writer.writerows(crop_rows)
    
    return

def znorm(img):
    return (img - np.mean(img)) / np.std(img)

'''
Let's collect the above operation into one function
Inputs: crop of the bounding box from the raw image array, 
        pixel size in nm, 
        diameter range, 
        max aspect ratio
Returns: diameter in nm given as major / minor axis length,
         boolean for whether granule is valid
         reason for invalidity if not valid
'''

def mask_touches_border(mask):
    return (
        mask[0, :].any() or
        mask[-1, :].any() or
        mask[:, 0].any() or
        mask[:, -1].any()
    )

def diameter_from_mask_touching_border(img_mask, px_size=3.22):

    def find_zeros(polar_mask):
        zeros_row = []
        for r in polar_mask:
            zeros = np.where(r == 0)[0][0]
            zeros_row.append(zeros)

        return zeros_row

    polar_mask = warp_polar(
        img_mask,
        center=(img_mask.shape[0] / 2, img_mask.shape[1] / 2),
        radius=img_mask.shape[0] / 2,
        scaling="linear"
    )
    valid_angles = ~polar_mask[:, -5]
    selected_polar_mask = polar_mask[valid_angles, :]
    
    selected_zeros = find_zeros(selected_polar_mask)

    return np.median(selected_zeros) * px_size * 2

def evaluate_granule(img_arr, pixel_size_nm=3.22, diameter_range=(50, 300), aspect_ratio_max=1.25):
    
    # normalize
    img = znorm(img_arr)
    # img = median(img, disk(3))
    img = gaussian(img, sigma=1, preserve_range=True)
    
    # thresholding
    img_otsu = img < threshold_otsu(img)
    
    # morphological operations
    img_mask = remove_small_holes(img_otsu)
    img_mask = remove_small_objects(img_mask, max_size=50)
    img_mask = closing(img_mask, disk(3))
    
    mask_touching_border = mask_touches_border(img_mask)
    print(f"mask_touches_border: {mask_touching_border}")

    rejection_reason = "N.A."
    quality_check = True
    if not mask_touching_border:
        # labeling and region properties
        lab = label(img_mask)
        props = regionprops(lab)
        
        if len(props) == 0:
            return None  # No objects found
        
        obj = props[0]
        
        major = obj.axis_major_length
        minor = obj.axis_minor_length
        diameter_mean_axes = (major + minor) / 2
        
        # check the quality of the granule based on diameter and aspect ratio
        # Check diameter range and aspect ratio
        quality_check = True
        if not (diameter_range[0] <= diameter_mean_axes * pixel_size_nm <= diameter_range[1]):
            quality_check = False  # Diameter out of range
            rejection_reason = "Diameter out of range"
            return img_mask, diameter_mean_axes * pixel_size_nm, quality_check, rejection_reason

        # check aspect ratio only if diameter is within range
        aspect_ratio = major / minor if minor != 0 else float('inf')
        if aspect_ratio > aspect_ratio_max:
            quality_check = False  # Aspect ratio too high
            rejection_reason = "Shape too thin"
        
        return img_mask, diameter_mean_axes * pixel_size_nm, quality_check, rejection_reason
    
    else:
        diameter_median = diameter_from_mask_touching_border(img_mask, pixel_size_nm)
        if not (diameter_range[0] <= diameter_median <= diameter_range[1]):
            quality_check = False
            rejection_reason = "Diameter out of range"
        else:
            quality_check = True
        
        return img_mask, diameter_median, quality_check, rejection_reason
    


'''
Unused functions below
'''
def _get_shape_bbox(shape):
    """Return integer bounding box values for a rectangle-like ROI shape, or None."""
    if not all(hasattr(shape, attr) for attr in ("getX", "getY", "getWidth", "getHeight")):
        return None

    try:
        x = int(round(float(unwrap(shape.getX()))))
        y = int(round(float(unwrap(shape.getY()))))
        width = int(round(float(unwrap(shape.getWidth()))))
        height = int(round(float(unwrap(shape.getHeight()))))
    except (TypeError, ValueError, OverflowError):
        return None

    if width <= 0 or height <= 0:
        return None

    return x, y, width, height

def _save_roi_crops_for_image(image_path: Path, image_id: int, rois, crops_dir: Path):
    """Save bounding-box crops for all rectangle-like shapes on an image."""
    if not rois:
        return 0

    with Image.open(image_path) as img:
        img_width, img_height = img.size

    saved_count = 0
    with Image.open(image_path) as img:
        for roi_idx, roi in enumerate(rois):
            for shape_idx, shape in enumerate(roi.copyShapes()):
                bbox = _get_shape_bbox(shape)
                if bbox is None:
                    continue

                x, y, width, height = bbox
                scale = 4
                x1 = max(0, min(int(round(x / scale)), img_width - 1))
                y1 = max(0, min(int(round(y / scale)), img_height - 1))
                x2 = max(x1 + 1, min(int(round((x + width) / scale)), img_width))
                y2 = max(y1 + 1, min(int(round((y + height) / scale)), img_height))

                crop = img.crop((x1, y1, x2, y2))
                output_name = f"{image_id}_{roi_idx}_{shape_idx}_{x1}_{y1}_{x2}_{y2}.png"
                output_path = crops_dir / output_name
                crop.save(output_path)
                saved_count += 1

    return saved_count


def _print_shape_info(roi):
    
    print("ROI ID:", unwrap(roi.getId()))
    print("ROI name:", unwrap(roi.getName()))
    print("ROI description:", unwrap(roi.getDescription()))
    print("Number of shapes in ROI: ", len(roi.copyShapes()))

    for shape in roi.copyShapes():
        print("  shape type:", type(shape))
        print("  shape id:", unwrap(shape.getId()))
        print("  shape text:", unwrap(shape.getTextValue()) if hasattr(shape, "getTextValue") else None)
        print("  shape description:", unwrap(shape.getDescription()) if hasattr(shape, "getDescription") else None)