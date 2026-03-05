import argparse
import shutil
import config
from datetime import datetime 
from pathlib import Path
import distutils.util
from common import init, download_downscaled_image, class_to_color, get_class_name_from_id, add_common_args, read_config_from_file
from ccipy.utils.cci_logger import CCILogger
from ccipy.utils.cci_colors import rgb_color
from ccipy.yolo_utils.cci_yolo_wrapper import CCIYoloWrapper
from ccipy.utils.roi_geometry import RoiRectangle
from ccipy.omero.omero_getter_ctx import OmeroGetterCtx
from ccipy.omero.geometry_to_roi import geometry_to_roi_shape
from ccipy.omero.omero_roi_helpers import remove_rois_from_dataset_by_name, remove_rois_from_dataset_by_description

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
        type=distutils.util.strtobool,
        required=False,
        default=False,
        help="Filter boxes at the border of the image"
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
        type=distutils.util.strtobool,
        required=False,
        default=False,
        help="Remove existing ROIs from the dataset, using the value from config.AI_ROI_NAME to identify them"
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

    if remove_rois:
        remove_rois_from_dataset_by_name(connection, dataset_id, config.AI_ROI_NAME)
        remove_rois_from_dataset_by_description(connection, dataset_id, config.AI_ROI_DESCRIPTION)
        
    datafiles_path = Path("datafiles")

    shutil.rmtree(datafiles_path, ignore_errors=True)
    images_dir = Path("datafiles/images") / str(dataset_id)
    images_dir.mkdir(exist_ok=True, parents=True)

    my_img_size = 512

    yolo_wrapper = CCIYoloWrapper()
    yolo_wrapper.load_model(weights_path=model_dir)

    with OmeroGetterCtx(connection) as getter:

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M")   # e.g., 2026-02-25
        output_file = "annotations_" + date_str + ".csv"

        csv_file = open(output_file, "w")
        csv_file.write("dataset,")
        csv_file.write("image name,")
        csv_file.write("image id,")
        csv_file.write("class name,")
        csv_file.write("class id,")
        csv_file.write("color,")
        csv_file.write("width,")
        csv_file.write("height\n")
        
    #    for dataset_id in dataset_ids:
        for img_id in getter.get_image_ids_from_dataset(dataset_id):

            img = getter.conn.get_image(img_id)
            img_name = img.getName()
            if not img_name.endswith("ome.tif"):
                CCILogger.info(f"Skipping image {img_id} with name {img_name} as it is not an OME-TIF.")
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
                if conf < confidence_threshold:
                    CCILogger.warning(f"Skipping box with confidence {conf:.4f} below threshold {confidence_threshold}")
                    continue
                xyxyn = box.xyxyn[0].tolist()  # [x1, y1, w, h] normalized
                class_name = get_class_name_from_id(cls_id)
                r,g,b = class_to_color(cls_id)
                color = rgb_color(r, g, b)
                rect = RoiRectangle.from_normalized_xyxy(xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], img_width, img_height, color, class_name + f" ({conf:.2f})")

                #filter this rect if it is at the border of the image
                if filter_border and (rect.x <= border_width or rect.y <= border_width or rect.x + rect.width >= img_width - border_width or rect.y + rect.height >= img_height - border_width):
                    CCILogger.warning(f"Skipping box at image border: {rect.x}, {rect.y}, {rect.width}, {rect.height}")
                    continue
                
                roi_shape = geometry_to_roi_shape(rect)
                shapes.append(roi_shape)
                CCILogger.info(f"Class ID: {cls_id}, Confidence: {conf:.4f}, Box: {xyxyn}, Color: ({r}, {g}, {b})")
                
                csv_file.write(f"{dataset_id},")
                csv_file.write(f"{img_name},")
                csv_file.write(f"{img_id},")
                csv_file.write(f"{class_name},")
                csv_file.write(f"{cls_id},")
                csv_file.write(f"{color},")
                csv_file.write(f"{rect.width},")
                csv_file.write(f"{rect.height}\n")

            getter.set_rois_on_image(img_id, shapes, config.AI_ROI_NAME, config.AI_ROI_DESCRIPTION)
        
        csv_file.close()
        connection.attach_file_to_dataset(dataset_id, output_file, description=f"Annotations for dataset {dataset_id}", mimetype="text/plain")
                
    CCILogger.info("Done.")
    connection.close()
        
if __name__ == "__main__":
    main()
