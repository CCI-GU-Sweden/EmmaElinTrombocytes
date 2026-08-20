import argparse
import shutil
import config
from datetime import datetime 
from pathlib import Path
from PIL import Image
import csv
import tifffile as tiff
import numpy as np
# import distutils.util
from common import (init, 
                    download_downscaled_image, 
                    class_to_color, 
                    get_class_name_from_id, 
                    add_common_args, 
                    read_config_from_file,
                    _unwrap_omero_id,
                    _unwrap_omero_string,
                    crop_from_predicted_bbox,
                    get_crops_and_rois_from_dataset,
                    evaluate_granule
                    )
from ccipy.utils.cci_logger import CCILogger
from ccipy.utils.cci_colors import rgb_color
from ccipy.yolo_utils.cci_yolo_wrapper import CCIYoloWrapper
from ccipy.utils.roi_geometry import RoiRectangle
from ccipy.omero.omero_getter_ctx import OmeroGetterCtx
from ccipy.omero.geometry_to_roi import geometry_to_roi_shape
from omero.rtypes import unwrap
from omero.rtypes import rstring
import time

def rectangles_intersect(rect1, rect2):
    """Check if two RoiRectangle objects overlap (any intersection counts as within)."""
    return not (rect1.x + rect1.width < rect2.x or 
                rect2.x + rect2.width < rect1.x or 
                rect1.y + rect1.height < rect2.y or 
                rect2.y + rect2.height < rect1.y)

def _get_roi_owner_id(roi):
    details = roi.getDetails()
    owner = details.getOwner() if details is not None else None
    owner_id = owner.getId() if owner is not None else None
    return _unwrap_omero_id(owner_id)

def _remove_shapes(roi, us, roi_name, roi_description):
    roi_current_name = unwrap(roi.getName()) if hasattr(roi, "getName") else None

    if roi_current_name == roi_name or roi_current_name == roi_description:
        deleted_count = len(roi.copyShapes())
        us.deleteObject(roi)
        return deleted_count

    return 0

def remove_owned_ai_rois_from_dataset(omero_conn, dataset_id, roi_name, roi_description):
    current_user_id = _unwrap_omero_id(omero_conn.get_user_id())
    if current_user_id is None:
        CCILogger.warning("Could not resolve current OMERO user id. Skipping ROI removal for safety.")
        return 0

    print(f"roi_name: {roi_name}, roi_description: {roi_description}")
    total_removed = 0
    with OmeroGetterCtx(omero_conn) as getter:
        image_ids = list(getter.get_image_ids_from_dataset(dataset_id))
        print(f"Number of image IDs = {len(image_ids)}")
        us = omero_conn.get_update_service()

        # test_ids = [24429, 24427, 24382, 24430, 24381]
        for image_id in image_ids:
            # if image_id not in test_ids:
            #     continue
            # print(f"\nImage ID {image_id}")
            rois = getter.get_rois_for_image(image_id)
            removed_on_image = 0
            for roi in rois:
                removed_on_image += _remove_shapes(roi, us, roi_name, roi_description)

            if removed_on_image > 0:
                CCILogger.info(f"Removed {removed_on_image} own AI ROIs from image ID {image_id}.")
            total_removed += removed_on_image

    CCILogger.info(
        f"Removed {total_removed} own AI ROIs from dataset ID {dataset_id} for user ID {current_user_id}."
    )
    return total_removed
    
def get_pixel_size_and_unit(omero_conn, dataset_id):
    with OmeroGetterCtx(omero_conn) as getter:
        image_ids = list(getter.get_image_ids_from_dataset(dataset_id))
        print(f"Number of image IDs = {len(image_ids)}")
        us = omero_conn.get_update_service()
        
        for image_id in image_ids[:1]:
            # getter.download_original_image_file(image_id, images_dir)
            print(f"\nIMAGE ID {image_id}")
            img = omero_conn.get_image(image_id)
            pixels = img.getPrimaryPixels()._obj
            px_x = pixels.getPhysicalSizeX()._value
            px_unit = pixels.getPhysicalSizeX().getUnit()

    return px_x, _unwrap_omero_string(px_unit)

def main():
    parser = argparse.ArgumentParser(description="Process a list of numbers and a connection token.")

    # Add arguments
    parser.add_argument(
        "--dataset",
        type=int,
        required=False,
        help="ID of the OMERO dataset to annotate"
    )
    
    parser.add_argument(
        "--model_dir",
        type=str,
        required=False,
        help="Relative path to model directory"
    )
    
    parser.add_argument(
        "--filter_border",
        action="store_true",
        required=False,
        default=False,
        help="Filter boxes at the border of the image",
    )
    
    parser.add_argument(
        "--border_width",
        type=int,
        required=False,
        default=0,
        help="Boxes closer to the border than this value (in pixels) will be filtered if --filter_border is set"
    )
    
    parser.add_argument(
        "--remove_rois",
        action="store_true",
        required=False,
        default=False,
        help="Remove existing ROIs from the dataset, using the value from config.AI_ROI_NAME to identify them",
    )
    
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        required=False,
        default=0.0,
        help="Set confidence threshold for predictions. Dont include boxes with confidence below this threshold."
    )

    add_common_args(parser)

    # Parse arguments
    args = parser.parse_args()

    if args.token is None:
        CCILogger.error("Token is required. Please provide a token using the --token argument.")
        exit(1)

    dataset_id = args.dataset        
    token = args.token
    model_dir = ""
    if args.model_dir is not None:
        model_dir = Path(args.model_dir)
    filter_border = args.filter_border
    remove_rois = args.remove_rois
    border_width = args.border_width
    group = args.group if args.group else None
    use_test_host = args.use_test_host
    confidence_threshold = args.confidence_threshold
    min_granule_diameter = None
    max_granule_diameter = None

    if args.config_name is not None:
        config_from_file = read_config_from_file("annotate", args.config_name)
        if config_from_file is not None:
            common_config, config_from_file = config_from_file
            
            #first read common config, then overwrite with specific config values if they exist
            group = common_config.get("omero_group", None)
            use_test_host = common_config.get("use_test_host", use_test_host)
            
            #read specific config values
            dataset_id = config_from_file.get("dataset_id", dataset_id)
            model_dir = Path(config_from_file.get("model_dir", model_dir))
            filter_border = config_from_file.get("filter_border", filter_border)
            remove_rois = config_from_file.get("remove_rois", remove_rois)
            border_width = config_from_file.get("border_width", border_width)
            confidence_threshold = config_from_file.get("confidence_threshold", confidence_threshold)
            min_granule_diameter = config_from_file.get("min_granule_diameter", min_granule_diameter)
            max_granule_diameter = config_from_file.get("max_granule_diameter", max_granule_diameter)
            granule_diameter_range = (min_granule_diameter, max_granule_diameter)

            #overrides for common config with specific config values if they exist
            group = config_from_file.get("omero_group", group)
            use_test_host = config_from_file.get("use_test_host", use_test_host)
        else:
            CCILogger.warning(f"Could not read config {args.config_name} from file config file.")
            exit(1)

    if group is None:
        CCILogger.error("Group is required. Please provide a group using the --group argument or in the config file.")
        exit(1)

    print("Use test host:", use_test_host)
    print("Token:", token)
    print("Group:", group)

    print("Dataset:", dataset_id)

    print("Remove existing ROIs:", remove_rois)
    print("ROI name:", config.AI_ROI_NAME)
    print("ROI description:", config.AI_ROI_DESCRIPTION)
    
    
    print("Model dir:", model_dir)
    print("Filter border boxes:", filter_border)
    print("Border width:", border_width)
    print("Confidence threshold:", confidence_threshold)
        
    # Ask for confirmation
    confirm = input("\nIs this correct? (Press 'y' to confirm, any other key to exit): ").strip().lower()
    if confirm != 'y':
        print("Exiting. Please check your input and try again.")
        return

    session_token = token
    connection = init(session_token, group, use_test_host=use_test_host)

    px_size, px_unit = get_pixel_size_and_unit(connection, dataset_id)
    if px_unit.lower() == "nanometer":
        px_unit = "nm"

    print(f"Pixel size: {px_size} {px_unit}")
    # return

    extract_granule_crops = False # Set to True if you want to extract granule crops and rois instead of running inference
    # script will exit after saving the crops from prior rois
    if extract_granule_crops:
        crops_dir = Path("datafiles/crops") / str(dataset_id)
        shutil.rmtree(crops_dir, ignore_errors=True)
        crops_dir.mkdir(exist_ok=True, parents=True)
        get_crops_and_rois_from_dataset(connection, dataset_id, crops_dir, num_crops=None)
        return

    if remove_rois:
        print("Removing existing ROIs...\n")
        remove_owned_ai_rois_from_dataset(connection, dataset_id, config.AI_ROI_NAME, config.AI_ROI_DESCRIPTION)
    
    print('Done checking ROI removal')
    # return

    datafiles_path = Path("datafiles")
    shutil.rmtree(datafiles_path, ignore_errors=True)
    datafiles_path.mkdir(exist_ok=True, parents=True)
    images_dir = Path("datafiles/images") / str(dataset_id)
    images_dir.mkdir(exist_ok=True, parents=True)

    my_img_size = config.YOLO_IMAGE_SIZE

    yolo_wrapper = CCIYoloWrapper()
    yolo_wrapper.load_model(weights_path=model_dir)

    inference_start_time = time.time()
    with OmeroGetterCtx(connection) as getter:

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M")   # e.g., 2026-02-25
        output_file = "annotations_" + date_str + ".csv"

        if remove_rois:
            csv_file = open(output_file, "w")

            csv_file.write("dataset,")
            csv_file.write("image name,")
            csv_file.write("image id,")
            csv_file.write("pixel size,")
            csv_file.write("pixel unit,")
            csv_file.write("class name,")
            csv_file.write("class id,")
            csv_file.write("color,")
            csv_file.write("conf,")
            csv_file.write("x topleft pixels,")
            csv_file.write("y topleft pixels,")
            csv_file.write("width pixels,")
            csv_file.write("height pixels,")
            csv_file.write(f"object diameter {px_unit},")
            csv_file.write("object validity,")
            csv_file.write("rejection reason\n")
            
        
    #    for dataset_id in dataset_ids:
        image_ids = list(getter.get_image_ids_from_dataset(dataset_id))
        num_imgs = len(image_ids)
        for img_id in image_ids:

            img = getter.conn.get_image(img_id)
            img_name = img.getName()
            if not img_name.endswith("ome.tif"):
                CCILogger.info(f"Skipping image {img_id} with name {img_name} as it is not an OME-TIF.")
                continue

            img_path, img_width, img_height, px_x, px_y, px_unit = download_downscaled_image(connection,  img_id, images_dir, img_size=my_img_size)
            CCILogger.info(f"Downloaded and downscaled image {img_path} with size {img_width}x{img_height}")

            print(f"image path: {img_path}")
            print(f"image width and height: {(img_width, img_height)}")
            print(f"pixel sizes x and y: {(px_x, px_y)}")
            print(f"pixel spatial unit: {px_unit}")

            if remove_rois: # if the previous rois were deleted, then we redo the inference and annotation
                pred = yolo_wrapper.predict(img=img_path)
                if not pred or len(pred) == 0:
                    CCILogger.warning(f"No prediction returned for image {img_path}")
                    continue
                
                CCILogger.info(f"Prediction result is for : {img_path}")
                boxes = pred[0].boxes

                # get the pixel sizes from the original image and the original image as an array
                pixels = img.getPrimaryPixels()._obj
                px_x = pixels.getPhysicalSizeX()._value
                px_y = pixels.getPhysicalSizeY()._value
                pixels = img.getPrimaryPixels()

                img_arr = pixels.getPlane(
                    theZ=0,
                    theC=0,
                    theT=0
                )
                
                # First pass: collect all cell rectangles
                cell_rects = []
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == 0:  # Cell class
                        conf = float(box.conf[0])
                        if conf < confidence_threshold:
                            continue
                        xyxyn = box.xyxyn[0].tolist()
                        rect = RoiRectangle.from_normalized_xyxy(xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], img_width, img_height, None, "Cell")
                        if not (filter_border and (rect.x <= border_width or rect.y <= border_width or rect.x + rect.width >= img_width - border_width or rect.y + rect.height >= img_height - border_width)):
                            cell_rects.append(rect)
                
                # Second pass: process all boxes
                shapes = []
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf < confidence_threshold:
                        CCILogger.warning(f"Skipping box with confidence {conf:.4f} below threshold {confidence_threshold}")
                        continue
                    xyxyn = box.xyxyn[0].tolist()  # [x1, y1, w, h] normalized
                    class_name = get_class_name_from_id(cls_id)
                    r,g,b = class_to_color(cls_id)
                    color = rgb_color(r, g, b)
                    rect = RoiRectangle.from_normalized_xyxy(xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], img_width, img_height, color, class_name + f" ({conf:.2f})")
                    
                    if class_name == "Cell":
                        object_diameter = 0.5 * (rect.width + rect.height) * px_size  # average of width and height in pixels
                        quality_check = True
                        rejection_reason = "N.A."
                    # crop the image into a new array using the predicted bounding box and evaluate granule quality
                    if class_name == "Granule":
                        crop_arr = crop_from_predicted_bbox(img_arr, rect)
                        _, object_diameter, quality_check, rejection_reason = evaluate_granule(crop_arr, pixel_size_nm=px_size, diameter_range=granule_diameter_range)


                    #filter this rect if it is at the border of the image
                    if filter_border and (rect.x <= border_width or rect.y <= border_width or rect.x + rect.width >= img_width - border_width or rect.y + rect.height >= img_height - border_width):
                        CCILogger.warning(f"Skipping box at image border: {rect.x}, {rect.y}, {rect.width}, {rect.height}")
                        continue
                    
                    # Filter granules: keep only if they overlap with at least one cell
                    if cls_id in [1, 2, 3]:  # Granule classes
                        if not cell_rects:
                            CCILogger.warning(f"Skipping granule (class {cls_id}) because no cells were detected in image")
                            continue
                        if not any(rectangles_intersect(rect, cell_rect) for cell_rect in cell_rects):
                            CCILogger.info(f"Skipping granule (class {cls_id}) because it is not within any cell")
                            continue
                    
                    roi_shape = geometry_to_roi_shape(rect)
                    shape_class = roi_shape.getTextValue()._val if hasattr(roi_shape, "getTextValue") else None
                    # if not shape_class:
                        # roi_shape.setTextValue(rstring(config.AI_ROI_NAME)) # added in case L373 doesn't do it

                    class_text = rstring(class_name + f" ({conf:.2f})")
                    if not quality_check:
                        class_text = rstring(class_name + f" ({conf:.2f}) - REJECTED")
                    roi_shape.setTextValue(class_text)
                    
                    shapes.append(roi_shape)
                    CCILogger.info(f"Class ID: {cls_id}, Confidence: {conf:.4f}, Box: {xyxyn}, Color: ({r}, {g}, {b})")
                            
                    # csv_file.write("dataset,")
                    # csv_file.write("image name,")
                    # csv_file.write("image id,")
                    # csv_file.write("pixel size,")
                    # csv_file.write("pixel unit,")
                    # csv_file.write("class name,")
                    # csv_file.write("class id,")
                    # csv_file.write("color,")
                    # csv_file.write("conf,")
                    # csv_file.write("x topleft pixels,")
                    # csv_file.write("y topleft pixels,")
                    # csv_file.write("width pixels,")
                    # csv_file.write("height pixels,")
                    # csv_file.write(f"object diameter {px_unit},")
                    # csv_file.write("object validity,")
                    # csv_file.write("rejection reason\n")

                    csv_file.write(f"{dataset_id},")
                    csv_file.write(f"{img_name},")
                    csv_file.write(f"{img_id},")
                    csv_file.write(f"{px_size},")
                    csv_file.write(f"{px_unit},")
                    csv_file.write(f"{class_name},")
                    csv_file.write(f"{cls_id},")
                    csv_file.write(f"{color},")
                    csv_file.write(f"{conf},")
                    csv_file.write(f"{rect.x},")
                    csv_file.write(f"{rect.y},")
                    csv_file.write(f"{rect.width},")
                    csv_file.write(f"{rect.height},")
                    csv_file.write(f"{object_diameter},")
                    csv_file.write(f"{quality_check},")
                    csv_file.write(f"{rejection_reason}\n")
                    
            
                getter.set_rois_on_image(img_id, shapes, config.AI_ROI_NAME, config.AI_ROI_DESCRIPTION)
        
        csv_file.close()
        connection.attach_file_to_dataset(dataset_id, output_file, description=f"Annotations for dataset {dataset_id}", mimetype="text/plain")
                
    CCILogger.info("Done.")
    elapsed = time.time() - inference_start_time
    CCILogger.info(f"Inference and annotation on {num_imgs} images took {elapsed:.2f} seconds.")
    CCILogger.info(f"Average time per image: {elapsed/num_imgs:.2f} seconds.")
    connection.close()
        
if __name__ == "__main__":
    main()