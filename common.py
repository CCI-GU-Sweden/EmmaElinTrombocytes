from pathlib import Path
import os
from ccipy.utils.cci_logger import CCILogger
from ccipy.omero.cci_omero_connection import OmeroConnection
from ccipy.omero.omero_getter_ctx import OmeroGetterCtx
from ccipy.yolo_utils.vectors_from_geometries import geometries_to_vectors_normalized, save_vectors_to_txt
from ccipy.omero.omero_colors import omero_rint_to_rgb,omero_rint_to_rgba
from ccipy.utils.roi_geometry import RoiGeometry
from skimage.transform import resize
from PIL import Image
import shutil
import tifffile as tiff
import numpy as np

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

def init(session_token: str, session_group: str, use_test_host: bool = False) -> OmeroConnection:

    CCILogger.setup_logger("logfile.log", "omero_test")

    if use_test_host:
        connection = OmeroConnection(OMERO_TEST_HOST,OMERO_PORT,session_token)
    else:
        connection = OmeroConnection(OMERO_HOST,OMERO_PORT,session_token)
    
    connection.set_group_name_for_session(session_group)
    return connection

def color_stretch(img, low_pt=1, high_pt=99):
    img = img.astype(np.float32)

    lo = np.percentile(img, low_pt)
    hi = np.percentile(img, high_pt)

    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)

    stretched = (img - lo) / (hi - lo)

    return stretched


def downscale_data(image: Path, target_size: tuple[int, int]):
    if image.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        return None
    img = tiff.imread(image)
    original_size = img.shape[:2]
    if original_size != target_size:
        CCILogger.info(f"Color stetching image {image.name}")
        stretched = color_stretch(img)
        
        CCILogger.info(f"Downscaling image {image.name} from {original_size} to {target_size}")
        if len(img.shape) == 3:
            downscaled_img = resize(stretched, target_size + (img.shape[2],), anti_aliasing=True)
        else:
            downscaled_img = resize(stretched, target_size, anti_aliasing=True)
            
        downscaled_img = downscaled_img * 255.0

        downscaled_img = np.clip(downscaled_img, 0, 255).astype(np.uint8)
        #downscaled_img = (downscaled_img * 65535).astype(np.uint16)
        im = Image.fromarray(downscaled_img)
        img_name_png = str(image).replace(".ome.tiff",".png")
        im.save(img_name_png)

        #tiff.imwrite(img_name_tif, downscaled_img)
        os.unlink(image)
        return Path(img_name_png)
    else:
        CCILogger.info(f"Image {image.name} already at target size {target_size}, skipping downscale.")
    return image.stem

def download_downscaled_image(connection: OmeroConnection, img_id: int, images_dir: Path, img_size: int = 512) -> tuple[str, int, int]:
    with OmeroGetterCtx(connection) as getter:
        img = connection.get_image(img_id)
        img_width = img.getSizeX()
        img_height = img.getSizeY()
        getter.download_original_image_file(img_id, images_dir)
        new_img_name = downscale_data(images_dir / Path(img.getName()), (img_size, img_size))
        
        return new_img_name, img_width, img_height


# def download_and_downscale_image(connection: OmeroConnection, img_id, vectors_dir: Path, images_dir: Path, img_size: int = 512):

#     with OmeroGetterCtx(connection) as getter:
#         img = connection.get_image(img_id)
#         img_width = img.getSizeX()
#         img_height = img.getSizeY()
#         getter.download_original_image_file(img_id, images_dir)
#         new_img_name = downscale_data(images_dir / Path(img.getName()), (img_size, img_size))


# def download_images(connection: OmeroConnection, dataset_ids: list[int], vectors_dir: Path, images_dir: Path, img_size: int = 512):

#     with OmeroGetterCtx(connection) as getter:
#         for dataset_id in dataset_ids:
#             for img_id in getter.get_image_ids_from_dataset(dataset_id):
#                 download_downscaled_image(connection, img_id, images_dir, img_size)


def geometry_to_class_definitions(geometry: RoiGeometry) -> int:
    # Blue (rgb(0, 181, 255)) = Cell
    # Yellow (rgb(255, 255, 0)) = Granule
    # Purple (rgb(152, 0, 255) = Atypical granule
    # Red (rgb(255, 0, 0)) = Unclear if granule
    color = geometry.get_color()
    r, g, b, a = omero_rint_to_rgba(color)
    if r == 0 and g == 181 and b == 255:
        return 0
    elif r == 255 and g == 255 and b == 0:
        return 1
    elif r == 152 and g == 0 and b == 255:
        return 2
    elif r == 255 and g == 0 and b == 0:
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