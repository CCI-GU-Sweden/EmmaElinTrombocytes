import argparse
import csv
import io
import logging
import sys
import config
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import distutils.util
from omero.gateway import FileAnnotationWrapper
from omero.model import FileAnnotationI
from omero.rtypes import rstring
from common import init, load_inference_image_in_memory, class_to_color, get_class_name_from_id, add_common_args, read_config_from_file
from common import crop_from_predicted_bbox, evaluate_granule
from ccipy.utils.cci_logger import CCILogger
from ccipy.utils.cci_colors import rgb_color
from ccipy.yolo_utils.cci_yolo_wrapper import CCIYoloWrapper
from ccipy.utils.roi_geometry import RoiRectangle
from ccipy.omero.omero_getter_ctx import OmeroGetterCtx
from ccipy.omero.geometry_to_roi import geometry_to_roi_shape

def rectangles_intersect(rect1, rect2):
    """Check if two RoiRectangle objects overlap (any intersection counts as within)."""
    return not (rect1.x + rect1.width < rect2.x or
                rect2.x + rect2.width < rect1.x or
                rect1.y + rect1.height < rect2.y or
                rect2.y + rect2.height < rect1.y)

def configure_console_logger(token):
    class RedactToken(logging.Filter):
        def filter(self, record):
            record.msg = record.getMessage().replace(token, "[REDACTED]") if token else record.getMessage()
            record.args = ()
            return True

    logger = logging.getLogger("omero_inference")
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RedactToken())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    CCILogger._logger = logger


def attach_csv_to_dataset(connection, dataset_id, csv_text):
    """Upload a CSV directly from memory as a new dataset file annotation."""
    gateway = connection.get_omero_connection()
    dataset = connection.get_dataset(dataset_id)
    if dataset is None:
        raise ValueError(f"Dataset {dataset_id} is unavailable; CSV was not attached")
    name = f"annotations_{datetime.now():%Y-%m-%d_%H-%M-%S}_{uuid4().hex[:8]}.csv"
    data = csv_text.encode("utf-8")
    with io.BytesIO(data) as source:
        original = gateway.createOriginalFileFromFileObj(
            source, "/", name, len(data), mimetype="text/csv"
        )
    annotation = FileAnnotationI()
    annotation.setFile(original._obj)
    annotation.setDescription(rstring(f"Annotations for dataset {dataset_id}"))
    saved = connection.get_update_service().saveAndReturnObject(annotation)
    dataset.linkAnnotation(FileAnnotationWrapper(gateway, saved))
    CCILogger.info(f"Attached CSV {name} to dataset {dataset_id}.")


def pixel_size_nm(image):
    """Read per-image pixel sizes, rejecting units or anisotropy we cannot evaluate."""
    pixels = image.getPrimaryPixels()._obj
    sizes = []
    for length in (pixels.getPhysicalSizeX(), pixels.getPhysicalSizeY()):
        if length is None:
            raise ValueError("Granule quality checks require physical pixel sizes")
        unit = str(length.getUnit()).lower().split(".")[-1]
        factors = {"nanometer": 1, "nm": 1, "micrometer": 1000, "µm": 1000, "um": 1000}
        if unit not in factors:
            raise ValueError(f"Unsupported pixel size unit: {unit}")
        value = float(length.getValue()) * factors[unit]
        if value <= 0:
            raise ValueError("Pixel size must be positive")
        sizes.append(value)
    if abs(sizes[0] - sizes[1]) > max(sizes) * 1e-6:
        raise ValueError("Granule quality checks require equal X and Y pixel sizes")
    return sizes[0]


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
        "--confidence_threshold",
        type=float,
        required=False,
        default=0.0,
        help="Set confidence threshold for predictions. Dont include boxes with confidence below this threshold."
    )

    add_common_args(parser)

    # Parse arguments
    args = parser.parse_args()

    configure_console_logger(args.token)

    if args.token is None:
        CCILogger.error("Token is required. Please provide a token using the --token argument.")
        exit(1)

    dataset_id = args.dataset
    token = args.token
    model_dir = ""
    if args.model_dir is not None:
        model_dir = Path(args.model_dir)
    filter_border = args.filter_border
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
            if config_from_file.get("remove_rois", False):
                raise ValueError("remove_rois must be false: inference preserves existing ROIs")
            border_width = config_from_file.get("border_width", border_width)
            confidence_threshold = config_from_file.get("confidence_threshold", confidence_threshold)
            min_granule_diameter = config_from_file.get("min_granule_diameter")
            max_granule_diameter = config_from_file.get("max_granule_diameter")

            #overrides for common config with specific config values if they exist
            group = config_from_file.get("omero_group", group)
            use_test_host = config_from_file.get("use_test_host", use_test_host)
        else:
            CCILogger.warning(f"Could not read config {args.config_name} from file config file.")
            exit(1)

    if group is None:
        CCILogger.error("Group is required. Please provide a group using the --group argument or in the config file.")
        exit(1)

    if (min_granule_diameter is None) != (max_granule_diameter is None):
        raise ValueError("Both granule diameter limits must be provided in nanometers")
    quality_enabled = min_granule_diameter is not None
    if quality_enabled and not (0 <= min_granule_diameter <= max_granule_diameter):
        raise ValueError("Invalid granule diameter range")

    print("Use test host:", use_test_host)
    print("Group:", group)

    print("Dataset:", dataset_id)

    print("Existing ROIs will be preserved.")
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
    connection = init(session_token, group, use_test_host=use_test_host, init_log=False)

    try:
        my_img_size = config.YOLO_IMAGE_SIZE

        yolo_wrapper = CCIYoloWrapper()
        yolo_wrapper.load_model(weights_path=model_dir)

        csv_buffer = io.StringIO(newline="")
        csv_writer = csv.writer(csv_buffer)
        csv_writer.writerow(["dataset", "image name", "image id", "pixel size nm", "class name", "class id", "color", "conf", "x topleft pixels", "y topleft pixels", "width pixels", "height pixels", "object diameter nm", "object validity", "rejection reason"])

        with OmeroGetterCtx(connection) as getter:

            for img_id in getter.get_image_ids_from_dataset(dataset_id):

                img = getter.conn.get_image(img_id)
                img_name = img.getName()
                if not img_name.endswith("ome.tif"):
                    CCILogger.info(f"Skipping image {img_id}: not an OME-TIF.")
                    continue

                inference_image, img_arr = load_inference_image_in_memory(img, img_size=my_img_size)
                img_width, img_height = img.getSizeX(), img.getSizeY()
                pred = yolo_wrapper.predict(img=inference_image)
                if not pred or len(pred) == 0:
                    CCILogger.warning(f"No prediction returned for image {img_id}")
                    del pred, inference_image, img_arr
                    continue

                CCILogger.info(f"Processing predictions for image {img_id}")
                boxes = pred[0].boxes
                px_size = pixel_size_nm(img) if quality_enabled else None

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
                rows = []
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

                    # Filter granules: keep only if they overlap with at least one cell
                    if cls_id in [1, 2, 3]:  # Granule classes
                        if not cell_rects:
                            CCILogger.warning(f"Skipping granule (class {cls_id}) because no cells were detected in image")
                            continue
                        if not any(rectangles_intersect(rect, cell_rect) for cell_rect in cell_rects):
                            CCILogger.info(f"Skipping granule (class {cls_id}) because it is not within any cell")
                            continue

                    object_diameter, quality_check, rejection_reason = None, None, "N.A."
                    if quality_enabled and class_name == "Cell":
                        object_diameter = 0.5 * (rect.width + rect.height) * px_size
                        quality_check = True
                    elif quality_enabled and class_name == "Granule":
                        crop = crop_from_predicted_bbox(img_arr, rect)
                        result = evaluate_granule(crop, pixel_size_nm=px_size, diameter_range=(min_granule_diameter, max_granule_diameter))
                        del crop
                        if result is None:
                            quality_check, rejection_reason = False, "No object found"
                        else:
                            _, object_diameter, quality_check, rejection_reason = result

                    roi_shape = geometry_to_roi_shape(rect)
                    if quality_check is False:
                        roi_shape.setTextValue(rstring(class_name + f" ({conf:.2f}) - REJECTED"))
                    shapes.append(roi_shape)
                    rows.append([dataset_id, img_name, img_id, px_size, class_name, cls_id, color, conf, rect.x, rect.y, rect.width, rect.height, object_diameter, quality_check, rejection_reason])
                    CCILogger.info(f"Class ID: {cls_id}, Confidence: {conf:.4f}, Box: {xyxyn}, Color: ({r}, {g}, {b})")

                if shapes:
                    getter.set_rois_on_image(img_id, shapes, config.AI_ROI_NAME, config.AI_ROI_DESCRIPTION)
                    csv_writer.writerows(rows)
                del pred, boxes, inference_image, img_arr, shapes

        attach_csv_to_dataset(connection, dataset_id, csv_buffer.getvalue())
        csv_buffer.close()
        CCILogger.info("Done.")
    finally:
        connection.close()

if __name__ == "__main__":
    main()
