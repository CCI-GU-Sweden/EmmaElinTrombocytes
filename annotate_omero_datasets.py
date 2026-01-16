from common import init, download_downscaled_image, class_to_color, get_class_name_from_id
import os
import shutil
from pathlib import Path
from ccipy.utils.cci_logger import CCILogger
from ccipy.utils.cci_colors import Colors, rgb_color
from ccipy.omero.cci_omero_connection import OmeroConnection
from ccipy.omero.omero_getter_ctx import OmeroGetterCtx
from ccipy.yolo_utils.cci_yolo_wrapper import CCIYoloWrapper
from ccipy.utils.roi_geometry import RoiGeometry, RoiRectangle
from ccipy.omero.omero_colors import omero_rgb_to_rint
from ccipy.omero.geometry_to_roi import geometry_to_roi_shape



session_token = "0490be3c-1b12-4ff4-a9fa-e5e1cb47c02e"
connection = init(session_token,"Emma-Josefsson-Lab")
#connection = init(session_token,"cci-staff", True)

dataset_id = 1161 # Hens Berndtsson / 2025-11-06 / 1161

datafiles_path = Path("datafiles")

shutil.rmtree(datafiles_path, ignore_errors=True)
images_dir = Path("datafiles/images") / str(dataset_id)
images_dir.mkdir(exist_ok=True, parents=True)

my_img_size = 512

model_dir = Path("runs/detect/train2/weights/best.pt")
yolo_wrapper = CCIYoloWrapper()
yolo_wrapper.load_model(weights_path=model_dir)

with OmeroGetterCtx(connection) as getter:
#    for dataset_id in dataset_ids:
    for img_id in getter.get_image_ids_from_dataset(dataset_id):

        img = getter.conn.get_image(img_id)
        img_name = img.getName()
        if not img_name.endswith("ome.tiff"):
            CCILogger.info(f"Skipping image {img_id} with name {img_name} as it is not an OME-TIFF.")
            continue

        img_path, img_width, img_height = download_downscaled_image(connection,  img_id, images_dir, img_size=my_img_size)
        CCILogger.info(f"Downloaded and downscaled image {img_path} with size {img_width}x{img_height}")
        pred = yolo_wrapper.predict(img=img_path)
        if not pred or len(pred) == 0:
            CCILogger.warning(f"No prediction returned for image {img_path}")
            continue
        
        CCILogger.info(f"Prediction result is for : {img_path}")
        boxes = pred[0].boxes
        shapes = []
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxyn = box.xyxyn[0].tolist()  # [x1, y1, w, h] normalized
            #cls_id_str = str(cls_id)
            class_name = get_class_name_from_id(cls_id)
            r,g,b = class_to_color(cls_id)
            color = rgb_color(r, g, b)
            rect = RoiRectangle.from_normalized_xyxy(xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], img_width, img_height, color, class_name + f" ({conf:.2f})")
            roi_shape = geometry_to_roi_shape(rect)
            shapes.append(roi_shape)
            CCILogger.info(f"Class ID: {cls_id}, Confidence: {conf:.4f}, Box: {xyxyn}, Color: ({r}, {g}, {b})")
            
        getter.set_rois_on_image(img_id, shapes)
            #pred[0].save("output.png")
            
CCILogger.info("Done.")