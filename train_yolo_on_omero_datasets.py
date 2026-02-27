from pathlib import Path
import shutil
import argparse
import random
import os
from functools import partial
import config
import distutils.util
from common import init, geometry_to_class_definitions, download_downscaled_image, label_names, generate_date_directory, add_common_args, read_config_from_file
from ccipy.omero.cci_omero_connection import OmeroConnection
from ccipy.omero.omero_getter_ctx import OmeroGetterCtx
from ccipy.omero.roi_to_geometry import rois_to_geometries
from ccipy.utils.cci_logger import CCILogger
from ccipy.yolo_utils.vectors_from_geometries import geometries_to_vectors_normalized, save_vectors_to_txt
from ccipy.yolo_utils.create_training_data_set import create_training_set
from ccipy.yolo_utils.cci_yolo_wrapper import CCIYoloWrapper

dataset_ids = []
dataset_ids.append(1159)
dataset_ids.append(1214)
dataset_ids.append(1161)
dataset_ids.append(1226)


def create_vectors_from_rois(rois, vectors_dir: Path, image_name: str, orig_img_width: int, orig_img_height: int, skip_list = []):
    geometries = rois_to_geometries(rois)
    
    
    vectors = geometries_to_vectors_normalized(geometries, orig_img_width, orig_img_height, partial(geometry_to_class_definitions, skiplist=skip_list))
    save_vectors_to_txt(vectors, vectors_dir / Path(f"{image_name}.txt"))

def download_images_with_rois(connection: OmeroConnection, dataset_ids: list[int], vectors_dir: Path, images_dir: Path, img_size: int = 512, skip_list = []):

    with OmeroGetterCtx(connection) as getter:
        for dataset_id in dataset_ids:
            for img_id in getter.get_image_ids_from_dataset(dataset_id):
                rois = getter.get_rois_for_image(img_id)
                CCILogger.info(f"Number of ROIs for image {img_id}: {len(rois)}")
                if len(rois) > 0:
                    image_name, image_width, image_height = download_downscaled_image(connection, img_id, images_dir, img_size)
                    create_vectors_from_rois(rois, vectors_dir, image_name.stem, image_width, image_height, skip_list)
                    
def create_data_set(vectors_dir: Path, images_dir: Path, label_names: list[tuple[int, str]]):

    CCILogger.info("Creating data set...")
    create_training_set(vectors_dir, images_dir, Path("dataset"), label_names)


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
        type=distutils.util.strtobool,
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

    token = args.token

    group = "Emma-Josefsson-Lab"
    if args.group is not None:
        group = args.group

    use_test_host = False
    if args.use_test_host is not None:
        use_test_host = args.use_test_host

    model_save_dir = ""
    if args.config_name is not None:
        config_from_file = read_config_from_file("train", args.config_name)
        if config_from_file is not None:
            common_config, config_from_file = config_from_file
            datasets = config_from_file.get("datasets", datasets)
            epochs = config_from_file.get("epochs", epochs)
            patience = config_from_file.get("patience", patience)
            skip_dataset_creation = config_from_file.get("skip_dataset_creation", skip_dataset_creation)
            class_skip_list = config_from_file.get("skip_classes", class_skip_list)
            model_save_dir = config_from_file.get("model_save_dir", model_save_dir)
            group = common_config.get("omero_group", None)
            use_test_host = common_config.get("use_test_host", False)

            #token = config_from_file.get("token", token)

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

    save_path = model_save_dir / generate_date_directory()
    weights_path = res.save_dir / "weights"
    shutil.copytree(weights_path, save_path)
    output_file = save_path / "config_log.txt"

    with open(output_file, "w") as f:
        f.write(f"datasets: {datasets}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"patience: {patience}\n")
        f.write(f"skip_dataset_creation: {skip_dataset_creation}\n")
        f.write(f"skip_classes: {class_skip_list}\n")
        f.write("Token: SUPER_SECRET\n")

    test_yolo_model(images_dir)



if __name__ == "__main__":
    main()
