
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
- dataset_ids: datasets to annotate in order. List of unique positive integers, for example `[2401, 2402]`. A single `dataset_id` is also supported for older configs.
- model_dir: Model path to use for the annotation. Local! String <"cells_and_granules\\2026-03-04\\11-31\\best.pt">
- filter_border: Remove any object on the edge/border of the image. Bool <true>
- border_width: Remove any object close of the boerder by pixel. Int <10>
- remove_rois: When `true`, delete existing ROIs named exactly `from AI` from each eligible image before inference. When `false`, preserve all existing ROIs. Unnamed and differently named ROIs are always preserved.
- confidence_threshold: minimum threshold to take in account for the annotation. Float (0-1) <0.0>

config.py contains other important parameters (YOLO_IMAGE_SIZE, BATCH_SIZE...)

Inference reads one 4096×4096 single-plane grayscale image directly from OMERO pixels into memory and saves predictions to OMERO after each image. It builds a dataset CSV in memory and attaches it to the dataset in OMERO when the run completes if it contains prediction rows; it does not save images, the CSV, or a log file locally. Images with other dimensions or multiple Z, channel or time planes are logged and skipped, with their ROIs untouched. Other errors stop the run: completed images remain annotated, the current image may already have lost its old AI ROIs if `remove_rois` is true, no CSV is attached for the incomplete dataset, and later datasets are not processed. The training script still uses local image files.

Memory-only application processing does not prevent the operating system from writing memory to a page file or crash dump. Configure the inference computer accordingly for sensitive data. Any files left by earlier runs remain on disk and need separate handling.

Example:

Training
>> python train_yolo_on_omero_datasets.py --token TOKEN_FROM_OMERO --config_name train_cells_and_granules_last

Annotate:
>> python annotate_omero_datasets.py --token TOKEN_FROM_OMERO --config_name annotate_cells_and_granules

## Evaluate detections

Set `dataset_ids` in the `evaluate` configuration in `config.yaml`, then run:

```bash
python evaluate_omero_dataset_detections.py --token TOKEN_FROM_OMERO --config_name evaluate_cells_and_granules
```

Alternatively, evaluate a single dataset with `--dataset DATASET_ID --group GROUP --token TOKEN_FROM_OMERO` (add `--use_test_host true` for the test server). The script only reads OMERO data and never changes or removes ROIs. It evaluates all shapes in ROIs named exactly `from AI` as predictions and all shapes in unnamed ROIs as ground truth, including AI shapes marked rejected. Other named ROIs and colors are ignored. Cell and granule classes are identified by stroke RGB colors `(0, 181, 255)` and `(255, 255, 0)`, respectively, regardless of opacity. Shapes of those colors must be rectangles with positive dimensions; invalid shapes stop the run. All images in each dataset are evaluated. Matches are confined to the same image and class; Z and time plane values are ignored.

At IoU > 0.5, boxes are matched one-to-one, maximizing the number of matches and then their total IoU. Each matched pair is a true positive; unmatched predictions are false positives, and unmatched ground truth boxes are false negatives. Counts are summed over all images in a dataset before calculating its metrics; the cumulative metrics use counts summed over all selected datasets. As requested, TPR = TP / (TP + FP), FPR = FP / (TP + FP), and FNR = FN / (TP + FN). These TPR and FPR definitions are specific to this evaluation: TPR here equals precision, while recall is TP / (TP + FN). TN is `N/A` because no set of background candidates is defined. Ratios with zero denominators are also `N/A`.

Each run writes to a new timestamped folder under `evaluation/`, with two subfolders: `all_images` includes every image, while `annotated_images` excludes images without any shapes in unnamed ground-truth ROIs. Any unnamed shape counts for deciding whether an image is annotated, regardless of its color; the metrics still evaluate only cell and granule boxes. Each subfolder contains `<dataset_id>_evaluation_result.csv` and `<dataset_id>_confusion_matrix_<class>.png` for each class, plus `all_datasets.csv`, `all_datasets_cell.png`, `all_datasets_granule.png`, `dataset_ids.txt`, `missed_cases.csv` and `hallucinated_cases.csv`. The two case files cover all selected datasets and have one row per unmatched box, with dataset ID, image ID, case type and class. Multiple unmatched boxes of the same class in one image produce identical rows. The cumulative counts sum the dataset counts, so an image belonging to multiple selected datasets is counted once per dataset. All plots use the same color scale, including the cumulative plots and both image selections. Earlier results are left untouched.

The evaluation uses NumPy, SciPy and Matplotlib, which are already included in `requirements.txt`. To update an activated environment, run `python -m pip install -r requirements.txt`. The existing requirements include private Git repositories and a platform-specific Python 3.12 Linux wheel, so their existing access and platform requirements apply.
