
# EmmaElinTrombocytes

Detecting trombocytes with yolo and omero

## Summary

These scripts are dataset of EM data stored on Omero, and require to have Omero access (TOKEN_FROM_OMERO).

There is two scripts:

- train_yolo_on_omero_datasets.py -> Train Yolo from a list of dataset
- annotate_omero_datasets.py -> Infer a/multiple dataset on Omero

Both the images **and** annotations are stored on Omero. Different classes are encoded with different colors. These tools only work with the rectangular object selection of Omero. Omero does not support natively object segmentation.

## Installation

Requirement:

- ccipy suites (omero, yolo_utils) from the CCI GU Sweden repo. Follow the instructions for installation.

for GPU support, uninstall pytorch and ultralytics:
then:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics
```

Warning! check that the wheel cu124 is correct (in this case, window and CUDA higher or equal to 12.4). Check on their [website](https://pytorch.org/).

## Running

In the config.yaml, indicate the different part, either for inference or training.
Training:

- name: name to indicate as argument <train_cells_and_granules_last>
- model_save_dir: where the results model will be save. Str <"cells_and_granules">
- epochs: number of epoch (or steps) to run. Int <300>
- patience: number of epoch seeing no improvement to stop the training. Int <200>
- skip_classes: classes to skip. List of Int <[2, 3]>
- datasets: which omero's dataset to train on. List of Int <[1159, 1214, 1226, 1305, 1354, 1551, 1803, 1952, 2152, 2201, 2202, 2251, 2351, 2408, 2451]>
- skip_dataset_creation: either to skip or not the creation of the database. Bool <false>
- omero_group: Which omero lab/group the dataset owner is part of. String <"Emma-Josefsson-Lab">
- use_test_host: Use the test omero instance instead. Bool <false>
- ignore_rois_by_description: ROIS matching description from config.py variables will be ignored during training. Bool <false>
- ignore_rois_by_name: ROIS matching name from config.py variables will be ignored during training. Bool <true>

ignore_rois_by_xxx should be considered mutual exclusive! Both are optional though.

Annotate:

- name: name of the annotation config <annotate_cells_and_granules>
- dataset_id: dataset to annotate. Int <2401>
- model_dir: Model path to use for the annotation. Local! String <"cells_and_granules\\2026-03-04\\11-31\\best.pt">
- filter_border: Remove any object on the edge/border of the image. Bool <true>
- border_width: Remove any object close of the boerder by pixel. Int <10>
- remove_rois: Remove ROIS/Annotation matching name OR description from config.py - default if "from AI". Bool <false>
- confidence_threshold: minimum threshold to take in account for the annotation. Float (0-1) <0.0>

config.py contains other important parameters (YOLO_IMAGE_SIZE, BATCH_SIZE...)

Example:

Training
>> python train_yolo_on_omero_datasets.py --token TOKEN_FROM_OMERO --config_name train_cells_and_granules_last

Annotate:
>> python annotate_omero_datasets.py --token TOKEN_FROM_OMERO --config_name annotate_cells_and_granules
