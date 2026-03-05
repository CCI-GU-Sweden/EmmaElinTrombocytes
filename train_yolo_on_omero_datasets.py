import shutil
import argparse
import config
from pathlib import Path
from functools import partial
from distutils.util import strtobool
from common import init, geometry_to_class_definitions, download_downscaled_image, label_names, generate_date_directory, add_common_args, read_config_from_file
from ccipy.utils.cci_logger import CCILogger
from ccipy.omero.cci_omero_connection import OmeroConnection
from ccipy.omero.omero_getter_ctx import OmeroGetterCtx
from ccipy.omero.roi_to_geometry import rois_to_geometries
from ccipy.yolo_utils.vectors_from_geometries import geometries_to_vectors_normalized, save_vectors_to_txt
from ccipy.yolo_utils.create_training_data_set import create_training_set
from ccipy.yolo_utils.cci_yolo_wrapper import CCIYoloWrapper

dataset_ids = []

def create_vectors_from_rois(rois, vectors_dir: Path, image_name: str, orig_img_width: int, orig_img_height: int, skip_list = []):
    geometries = rois_to_geometries(rois)
    
    vectors = geometries_to_vectors_normalized(geometries, orig_img_width, orig_img_height, partial(geometry_to_class_definitions, skiplist=skip_list))
    save_vectors_to_txt(vectors, vectors_dir / Path(f"{image_name}.txt"))

def download_images_with_rois(connection: OmeroConnection, dataset_ids: list[int], vectors_dir: Path, images_dir: Path, img_size: int = 512, skip_list = [], ignore_rois_by_name = False, ignore_rois_by_description = False):

    with OmeroGetterCtx(connection) as getter:
        for dataset_id in dataset_ids:
            for img_id in getter.get_image_ids_from_dataset(dataset_id):
                rois = getter.get_rois_for_image(img_id)
                if len(rois) == 0:
                    CCILogger.info(f"No ROIs on image {img_id}. Skipping...")
                    continue
                if ignore_rois_by_name:
                    CCILogger.info("Ignoring ROIS by name...")
                    rois = [r for r in rois if r.getName().val != config.AI_ROI_NAME]
                if ignore_rois_by_description:
                    CCILogger.info("Ignoring ROIS by description...")
                    rois = [r for r in rois if r.getDescription().val != config.AI_ROI_DESCRIPTION]
                CCILogger.info(f"Number of ROIs for image {img_id}: {len(rois)}")
                image_name, image_width, image_height = download_downscaled_image(connection, img_id, images_dir, img_size)
                create_vectors_from_rois(rois, vectors_dir, image_name.stem, image_width, image_height, skip_list)
                    
def create_data_set(vectors_dir: Path, images_dir: Path, label_names: list[tuple[int, str]]):

    CCILogger.info("Creating data set...")
    create_training_set(vectors_dir, images_dir, Path("dataset"), label_names)

def main():
    parser = argparse.ArgumentParser(description="Process a list of numbers and a connection token.")

    # Add arguments
    parser.add_argument(
        "--datasets",
        nargs="+",  # Accepts one or more values
        type=int,  # Convert to float (use `int` if you want integers)
        required=False,
        help="List of numbers to process"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        help="Number of epochs to train"
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        required=False,
        help="Number of epochs with no improvement after which training will be stopped"
    )
    
    parser.add_argument(
        "--skip_dataset_creation",
        type=strtobool,
        required=False,
        default=False,
        help="Skip dataset creation and use existing dataset"
    )
    
    parser.add_argument(
        "--skip_classes",
        nargs="+",
        type=int,
        required=False,
        help="Skip classes in training"
    )
    
    parser.add_argument(
        "--ignore_rois_by_name",
        type=bool,
        required=False,
        default=False,
        help="Dont use rois with specific name in training (se config.py)"
    )
    
    parser.add_argument(
        "--ignore_rois_by_description",
        type=bool,
        required=False,
        default=False,
        help="Dont use rois with specific description in training (se config.py)"
    )
    
    
    add_common_args(parser)

    # Parse arguments
    args = parser.parse_args()

    # Access the arguments
    if args.datasets is None:
        datasets = dataset_ids
    else:
        datasets = args.datasets
        
    epochs = config.YOLO_EPOCHS
    if args.epochs is not None:
        epochs = args.epochs
        
    patience = config.YOLO_PATIENCE
    if args.patience is not None:
        patience = args.patience
        
    skip_dataset_creation = False
    if args.skip_dataset_creation is not None:
        skip_dataset_creation = args.skip_dataset_creation
        
    class_skip_list = []
    if args.skip_classes is not None:
        class_skip_list = args.skip_classes

    if args.token is None:
        CCILogger.error("Token is required. Please provide a token using the --token argument.")
        exit(1)

    ignore_rois_by_name = args.ignore_rois_by_name
    ignore_rois_by_description = args.ignore_rois_by_description
        
    token = args.token

    group = None
    if args.group is not None:
        group = args.group

    use_test_host = False
    if args.use_test_host is not None:
        use_test_host = args.use_test_host

    model_save_dir = "unnamed_model"
    if args.config_name is not None:
        config_from_file = read_config_from_file("train", args.config_name)
        if config_from_file is not None:
            common_config, config_from_file = config_from_file
            #first read common config, then overwrite with specific config values if they exist
            group = common_config.get("omero_group", None)
            use_test_host = common_config.get("use_test_host", use_test_host)
            
            #read specific config values
            datasets = config_from_file.get("datasets", datasets)
            epochs = config_from_file.get("epochs", epochs)
            patience = config_from_file.get("patience", patience)
            skip_dataset_creation = config_from_file.get("skip_dataset_creation", skip_dataset_creation)
            class_skip_list = config_from_file.get("skip_classes", class_skip_list)
            model_save_dir = config_from_file.get("model_save_dir", model_save_dir)
            ignore_rois_by_name = config_from_file.get("ignore_rois_by_name", ignore_rois_by_name)
            ignore_rois_by_description = config_from_file.get("ignore_rois_by_description", ignore_rois_by_description)
            
            #overrides for common config with specific config values if they exist
            group = config_from_file.get("omero_group", group)
            use_test_host = config_from_file.get("use_test_host", use_test_host)
        else:
            CCILogger.warning(f"Could not read config {args.config_name} from file config file.")
            exit(1)

    if group is None:
        CCILogger.error("Group is required. Please provide a group using the --group argument or in the config file.")
        exit(1)

    print("datasets:", datasets)
    print("epochs:", epochs)
    print("patience:", patience)
    print("skip_classes:", class_skip_list)
    print("skip_dataset_creation:", skip_dataset_creation)
    print("group:", group)
    print("use_test_host:", use_test_host)
    print("model_save_dir:", model_save_dir)
    print("Token:", token)
    
    # Ask for confirmation
    confirm = input("\nIs this correct? (Press 'y' to confirm, any other key to exit): ").strip().lower()
    if confirm != 'y':
        print("Exiting. Please check your input and try again.")
        return

    # Proceed if confirmed
    print("\nProceeding with the provided input...")

    my_img_size = config.YOLO_IMAGE_SIZE
    vectors_dir = Path("datafiles/vectors") 
    images_dir = Path("datafiles/images")

    session_token = token
    connection = init(session_token,"Emma-Josefsson-Lab")
    
    if not skip_dataset_creation:
        CCILogger.info("Creating training dataset from OMERO...")
        
        datafiles_path = Path("datafiles")

        shutil.rmtree(datafiles_path, ignore_errors=True)
        shutil.rmtree("dataset", ignore_errors=True)

        vectors_dir.mkdir(exist_ok=True, parents=True)
        images_dir.mkdir(exist_ok=True, parents=True)

        download_images_with_rois(connection, datasets, vectors_dir, images_dir, img_size=my_img_size, skip_list=class_skip_list)
        create_data_set(images_dir, vectors_dir, label_names)      

    else:
        CCILogger.info("Skipping training dataset creation, using existing dataset...")

    yolo_wrapper = CCIYoloWrapper()
    res = yolo_wrapper.train(data_set_file=Path("dataset/dataset.yaml"), epochs=epochs, patience=patience, batch=config.YOLO_BATCH_SIZE, image_size=my_img_size)

    run_epochs = yolo_wrapper.get_number_of_run_epochs()

    save_path = model_save_dir / generate_date_directory()
    weights_path = res.save_dir / "weights"
    shutil.copytree(weights_path, save_path)
    output_file = save_path / "config_log.txt"

    with open(output_file, "w") as f:
        f.write(f"datasets: {datasets}\n")
        f.write(f"given epochs: {epochs}\n")
        f.write(f"run epochs: {run_epochs+1} (if lower than given epochs, EarlyStopping was applied\n")
        f.write(f"patience: {patience}\n")
        f.write(f"skip_dataset_creation: {skip_dataset_creation}\n")
        f.write(f"skip_classes: {class_skip_list}\n")
        f.write(f"batch_size: {config.YOLO_BATCH_SIZE}\n")
        f.write(f"image_size: {my_img_size}\n")
        f.write(f"ROI name: {config.AI_ROI_NAME}\n")
        f.write(f"ROI description: {config.AI_ROI_DESCRIPTION}\n")
        f.write(f"ignore rois by name: {ignore_rois_by_name}\n")
        f.write(f"ignore rois by description: {ignore_rois_by_description}\n")
        f.write("Token: SUPER_SECRET\n")

    connection.close()

if __name__ == "__main__":
    main()
