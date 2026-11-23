"""Train YOLO on a dataset produced by prepare_yolo_dataset.py.

Edit the settings below and run this script in the project's Python environment.
For datasets without validation images, training runs for all EPOCHS, without
validation or early stopping; use weights/last.pt as the final model.
"""

DATASET_DIRECTORY = "yolo_dataset"
MODEL = "yolo11n.pt"  # Pretrained Ultralytics detection weights or a local .pt file
OUTPUT_DIRECTORY = "training_runs"
RUN_NAME = "cells_and_granules"
EPOCHS = 400
PATIENCE = 50  # Early stopping when a validation set exists
BATCH_SIZE = 16
IMAGE_SIZE = 1024
DEVICE = None  # None = automatic selection; e.g. "cpu" or 0
SEED = 42

from pathlib import Path


def main():
    dataset = Path(DATASET_DIRECTORY).expanduser().resolve()
    yaml_path = dataset / "data.yaml"
    train_dir = dataset / "train" / "images"
    val_dir = dataset / "val" / "images"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {yaml_path}")
    if not train_dir.is_dir() or not any(train_dir.glob("*.png")):
        raise ValueError(f"No training PNG images found in {train_dir}")
    if not val_dir.is_dir():
        raise ValueError(f"Validation directory not found: {val_dir}")
    has_validation = any(val_dir.glob("*.png"))

    import ultralytics
    from ultralytics import YOLO

    options = dict(
        data=str(yaml_path), epochs=EPOCHS, patience=PATIENCE,
        batch=BATCH_SIZE, imgsz=IMAGE_SIZE, device=DEVICE, seed=SEED,
        project=str(Path(OUTPUT_DIRECTORY).expanduser().resolve()),
        name=RUN_NAME, exist_ok=False,
    )
    model = YOLO(MODEL)
    if has_validation:
        print("Training with validation and early stopping.")
        model.train(**options)
    else:
        # These overrides match the trainer lifecycle in the project's pinned
        # ultralytics version. No test images are used during training.
        if ultralytics.__version__ != "8.3.233":
            raise RuntimeError("Training without validation requires ultralytics==8.3.233")
        from ultralytics.models.yolo.detect import DetectionTrainer

        class NoValidationTrainer(DetectionTrainer):
            def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
                if mode == "val":
                    dataset_path = self.data["train"]
                return super().get_dataloader(dataset_path, batch_size=batch_size, rank=rank, mode=mode)

            def validate(self):
                # Ultralytics calls validate on the last epoch even with val=False.
                return {}, -float(self.loss.detach().cpu())

            def final_eval(self):
                pass

        print("No validation images: training without validation or early stopping. Use last.pt.")
        options.update(val=False, patience=0, trainer=NoValidationTrainer)
        model.train(**options)
    print(f"Training output: {model.trainer.save_dir}")
    print(f"Final weights: {model.trainer.save_dir / 'weights' / 'last.pt'}")
    if has_validation:
        print(f"Best validation weights: {model.trainer.save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
