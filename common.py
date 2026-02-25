from pathlib import Path
from datetime import datetime
import os
from ccipy.utils.cci_logger import CCILogger
from ccipy.omero.cci_omero_connection import OmeroConnection
from ccipy.omero.omero_getter_ctx import OmeroGetterCtx
from ccipy.omero.omero_colors import omero_rint_to_rgba
from ccipy.utils.roi_geometry import RoiGeometry
from skimage.transform import resize
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

OMERO_HOST = "omero-cci-cli.gu.se"
OMERO_TEST_HOST = "omero-cli.test.gu.se"
OMERO_PORT = '4064'

def init(session_token: str, session_group: str, use_test_host: bool = False, init_log: bool = True) -> OmeroConnection:

    if init_log:
        CCILogger.setup_logger("logfile.log", "omero_test")

    if use_test_host:
        connection = OmeroConnection(OMERO_TEST_HOST,OMERO_PORT,session_token)
    else:
        connection = OmeroConnection(OMERO_HOST,OMERO_PORT,session_token)
    
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
        getter.download_original_image_file(img_id, images_dir)
        new_img_name = downscale_data(images_dir / Path(img.getName()), (img_size, img_size))
        
        return new_img_name, img_width, img_height


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