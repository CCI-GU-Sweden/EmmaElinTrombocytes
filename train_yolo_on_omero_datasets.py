from pathlib import Path
import shutil
# import tifffile as tiff
# import numpy as np

import random
import os
from common import init, downscale_data, geometry_to_class_definitions, download_downscaled_image, label_names
from ccipy.omero.cci_omero_connection import OmeroConnection
from ccipy.omero.omero_getter_ctx import OmeroGetterCtx
from ccipy.omero.roi_to_geometry import rois_to_geometries
# from ccipy.omero.omero_colors import omero_rint_to_rgb,omero_rint_to_rgba
from ccipy.utils.cci_logger import CCILogger
from ccipy.yolo_utils.vectors_from_geometries import geometries_to_vectors_normalized, save_vectors_to_txt
from ccipy.yolo_utils.create_training_data_set import create_training_set
from ccipy.yolo_utils.cci_yolo_wrapper import CCIYoloWrapper
# from ccipy.utils.roi_geometry import RoiGeometry




# def geometry_to_class_definitions(geometry: RoiGeometry) -> int:
#     # Blue (rgb(0, 181, 255)) = Cell
#     # Yellow (rgb(255, 255, 0)) = Granule
#     # Purple (rgb(152, 0, 255) = Atypical granule
#     # Red (rgb(255, 0, 0)) = Unclear if granule
#     color = geometry.get_color()
#     r, g, b, a = omero_rint_to_rgba(color)
#     if r == 0 and g == 181 and b == 255:
#         return 0
#     elif r == 255 and g == 255 and b == 0:
#         return 1
#     elif r == 152 and g == 0 and b == 255:
#         return 2
#     elif r == 255 and g == 0 and b == 0:
#         return 3
#     else:
#         CCILogger.warning(f"Color on geometry is not according to spec {r} {g} {b}")
#         raise ValueError("Wrong color")
    
def create_vectors_from_rois(rois, vectors_dir: Path, image_name: str, orig_img_width: int, orig_img_height: int):
    geometries = rois_to_geometries(rois)
    try:
        vectors = geometries_to_vectors_normalized(geometries, orig_img_width, orig_img_height, geometry_to_class_definitions)
        save_vectors_to_txt(vectors, vectors_dir / Path(f"{image_name}.txt"))
    except Exception:
        CCILogger.warning(f"Image {image_name} contains invalid colors...skipping")

# def download_downscaled_image(connection: OmeroConnection, img_id: int, images_dir: Path, img_size: int = 512) -> tuple[str, int, int]:
#     with OmeroGetterCtx(connection) as getter:
#         img = connection.get_image(img_id)
#         img_width = img.getSizeX()
#         img_height = img.getSizeY()
#         getter.download_original_image_file(img_id, images_dir)
#         new_img_name = downscale_data(images_dir / Path(img.getName()), (img_size, img_size))
        
#         return new_img_name, img_width, img_height

def download_images_with_rois(connection: OmeroConnection, dataset_ids: list[int], vectors_dir: Path, images_dir: Path, img_size: int = 512):

    with OmeroGetterCtx(connection) as getter:
        for dataset_id in dataset_ids:
            for img_id in getter.get_image_ids_from_dataset(dataset_id):
                rois = getter.get_rois_for_image(img_id)
                CCILogger.info(f"Number of ROIs for image {img_id}: {len(rois)}")
                if len(rois) > 0:
                    image_name, image_width, image_height = download_downscaled_image(connection, img_id, images_dir, img_size)
                    create_vectors_from_rois(rois, vectors_dir, image_name.stem, image_width, image_height)
                    
def create_data_set(vectors_dir: Path, images_dir: Path, label_names: list[tuple[int, str]]):

    CCILogger.info("Creating data set...")
    create_training_set(vectors_dir, images_dir, "dataset", label_names)


def test_yolo_model(images_dir: Path, img_idx = -1):
    
    model_dir = Path("runs/detect/train/weights/best.pt")
    yolo_wrapper = CCIYoloWrapper()
    yolo_wrapper.load_model(weights_path=model_dir)

    nr_imgs = len(os.listdir(images_dir))
    
    if img_idx != -1 and img_idx <= nr_imgs:
        test_image = images_dir / os.listdir(images_dir)[img_idx]    
    else:
        test_image = images_dir / os.listdir(images_dir)[random.randint(0, nr_imgs - 1)]

    pred = yolo_wrapper.predict(img=str(test_image))
    CCILogger.info(f"Prediction result is for : {str(test_image)}")

    pred[0].save("output.png")
    pred[0].show()


dataset_ids = []
dataset_ids.append(1159)
dataset_ids.append(1214)
session_token = "472cd8a3-faab-43b8-a9fc-936d480adef5"

connection = init(session_token,"Emma-Josefsson-Lab")


datafiles_path = Path("datafiles")

shutil.rmtree(datafiles_path, ignore_errors=True)
shutil.rmtree("dataset", ignore_errors=True)

vectors_dir = Path("datafiles/vectors")
vectors_dir.mkdir(exist_ok=True, parents=True)
images_dir = Path("datafiles/images")
images_dir.mkdir(exist_ok=True, parents=True)

my_img_size = 512

download_images_with_rois(connection, dataset_ids, vectors_dir, images_dir, img_size=my_img_size)
create_data_set(images_dir, vectors_dir, label_names)      

yolo_wrapper = CCIYoloWrapper()
res = yolo_wrapper.train(data_set_file=Path("dataset/dataset.yaml"), epochs=500, batch=16, image_size=my_img_size)

#model_dir = Path("runs/segment/train2/weights/best.pt")
test_yolo_model(images_dir)

