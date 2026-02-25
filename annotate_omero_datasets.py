from common import init, download_downscaled_image, class_to_color, get_class_name_from_id
import argparse
import shutil
from datetime import datetime 
import distutils.util
from pathlib import Path
from ccipy.utils.cci_logger import CCILogger
from ccipy.utils.cci_colors import rgb_color
from ccipy.yolo_utils.cci_yolo_wrapper import CCIYoloWrapper
from ccipy.utils.roi_geometry import RoiRectangle
from ccipy.omero.omero_getter_ctx import OmeroGetterCtx
from ccipy.omero.geometry_to_roi import geometry_to_roi_shape
from ccipy.omero.omero_roi_helpers import remove_rois_from_dataset

def main():
    parser = argparse.ArgumentParser(description="Process a list of numbers and a connection token.")

    # Add arguments
    parser.add_argument(
        "--dataset",
        type=int,
        required=True,
        help="List of numbers to process"
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Token for connections"
    )
    
    parser.add_argument(
        "--group",
        type=str,
        required=False,
        help="Group for the OMERO session"
    )
    
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
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
        help="Filter boxes at the border of the image"
    )
    
    parser.add_argument(
        "--remove_rois",
        type=distutils.util.strtobool,
        required=False,
        default=False,
        help="Remove existing ROIs from the dataset"
    )
    
    parser.add_argument(
        "--use_test_host",
        type=distutils.util.strtobool,
        required=False,
        default=False,
        help="Use test host"
    )
    
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        required=False,
        default=0.0,
        help="Set confidence threshold for predictions"
    )

    # Parse arguments
    args = parser.parse_args()

    dataset_id = args.dataset        
    token = args.token
    model_dir = Path(args.model_dir)
    filter_border = args.filter_border
    remove_rois = args.remove_rois
    border_width = args.border_width
    group = args.group if args.group else "Emma-Josefsson-Lab"
    use_test_host = args.use_test_host
    confidence_threshold = args.confidence_threshold

    print("dataset:", dataset_id)
    print("Token:", token)
    print("Group:", group)
    print("Model dir:", model_dir)
    print("Filter border boxes:", filter_border)
    print("Border width:", border_width)
    print("Remove existing ROIs:", remove_rois)
    print("Use test host:", use_test_host)
    print("Confidence threshold:", confidence_threshold)
        
    # Ask for confirmation
    confirm = input("\nIs this correct? (Press 'y' to confirm, any other key to exit): ").strip().lower()
    if confirm != 'y':
        print("Exiting. Please check your input and try again.")
        return

    session_token = token
    connection = init(session_token, group, use_test_host=use_test_host)

    if remove_rois:
        remove_rois_from_dataset(connection, dataset_id)

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
                #cls_id_str = str(cls_id)
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
                csv_file.write(f"{class_name},")
                csv_file.write(f"{cls_id},")
                csv_file.write(f"{color},")
                csv_file.write(f"{rect.width},")
                csv_file.write(f"{rect.height}\n")

            getter.set_rois_on_image(img_id, shapes)
        
        csv_file.close()
        connection.attach_file_to_dataset(dataset_id, output_file, description=f"Annotations for dataset {dataset_id}", mimetype="text/plain")
                
    CCILogger.info("Done.")
    
if __name__ == "__main__":
    main()
