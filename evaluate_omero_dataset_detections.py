"""Evaluate named AI and unnamed human rectangle ROIs without changing OMERO."""
import argparse
import csv
import math
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

IOU_THRESHOLD = 0.5
CLASS_COLORS = {"granule": (255, 255, 0), "cell": (0, 181, 255)}
OUTPUT_DIR = Path("evaluation")


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def match_counts(truth, predictions):
    """Maximize the number of valid one-to-one matches, then their total IoU."""
    if not truth or not predictions:
        return 0, len(predictions), len(truth)
    overlaps = np.array([[iou(a, b) for b in predictions] for a in truth])
    valid = overlaps > IOU_THRESHOLD
    # A valid match must outweigh any possible change in the total IoU.
    scores = valid * (min(len(truth), len(predictions)) + 1 + overlaps)
    rows, cols = linear_sum_assignment(scores, maximize=True)
    tp = int(valid[rows, cols].sum())
    return tp, len(predictions) - tp, len(truth) - tp


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics(counts):
    tp, fp, fn = counts
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    return [tp, fp, fn, None, precision, recall,
            ratio(2 * tp, 2 * tp + fp + fn), precision,
            ratio(fp, tp + fp), ratio(fn, tp + fn)]


def write_csv(path, results):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "tp", "fp", "fn", "tn", "precision", "recall", "f1_score", "tpr", "fpr", "fnr"])
        for name in CLASS_COLORS:
            writer.writerow([name] + ["N/A" if value is None else value for value in metrics(results[name])])


def write_cases(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset_id", "image_id", "case", "class"])
        writer.writerows(rows)


def plot_matrices(path, results, vmax, title, names):
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5), squeeze=False)
    for ax, name in zip(axes[0], names):
        tp, fp, fn = results[name]
        matrix = np.ma.array([[tp, fn], [fp, 0]], mask=[[False, False], [False, True]])
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        for row, col, value in [(0, 0, tp), (0, 1, fn), (1, 0, fp)]:
            ax.text(col, row, str(value), ha="center", va="center", color="white" if value > vmax / 2 else "black")
        ax.text(1, 1, "N/A", ha="center", va="center")
        ax.set(xticks=[0, 1], xticklabels=["Detected", "Missed"], yticks=[0, 1],
               yticklabels=["Object", "Background"], xlabel="Prediction", ylabel="Ground truth", title=name.capitalize())
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Count", shrink=0.8)
    fig.suptitle(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def collect_dataset(getter, dataset_id):
    from omero.rtypes import unwrap
    import config

    if getter.conn.get_dataset(dataset_id) is None:
        raise ValueError(f"Dataset {dataset_id} is unavailable")
    outcomes = {
        mode: {"counts": {name: [0, 0, 0] for name in CLASS_COLORS},
               "missed": [], "hallucinated": [], "images": 0}
        for mode in ("all_images", "annotated_images")
    }
    colors = {color: name for name, color in CLASS_COLORS.items()}
    for image_id in getter.get_image_ids_from_dataset(dataset_id):
        boxes = defaultdict(lambda: {"truth": [], "prediction": []})
        has_ground_truth = False
        for roi in getter.get_rois_for_image(image_id):
            name = unwrap(roi.getName())
            if name == config.AI_ROI_NAME:
                category = "prediction"
            elif not name:
                category = "truth"
            else:
                continue
            shapes = roi.copyShapes()
            if category == "truth" and shapes:
                has_ground_truth = True
            for shape in shapes:
                color = unwrap(shape.getStrokeColor())
                # OMERO stores RGBA as a signed 32-bit integer. Alpha is not
                # part of the class, and can differ between human and AI ROIs.
                rgb = tuple((color >> shift) & 255 for shift in (24, 16, 8)) if color is not None else None
                class_name = colors.get(rgb)
                if class_name is None:
                    continue
                try:
                    x, y, width, height = (float(unwrap(getattr(shape, method)())) for method in ("getX", "getY", "getWidth", "getHeight"))
                except (AttributeError, TypeError, ValueError) as exc:
                    raise ValueError(f"Image {image_id}: {class_name} ROI contains a non-rectangular or invalid shape") from exc
                if not all(math.isfinite(v) for v in (x, y, width, height)) or width <= 0 or height <= 0:
                    raise ValueError(f"Image {image_id}: {class_name} ROI has invalid rectangle dimensions")
                boxes[class_name][category].append((x, y, x + width, y + height))
        modes = ["all_images"]
        if has_ground_truth:
            modes.append("annotated_images")
        for mode in modes:
            outcomes[mode]["images"] += 1
        for class_name, groups in boxes.items():
            values = match_counts(groups["truth"], groups["prediction"])
            _, fp, fn = values
            for mode in modes:
                outcome = outcomes[mode]
                outcome["counts"][class_name] = [a + b for a, b in zip(outcome["counts"][class_name], values)]
                outcome["missed"].extend([(dataset_id, image_id, "missed", class_name)] * fn)
                outcome["hallucinated"].extend([(dataset_id, image_id, "hallucinated", class_name)] * fp)
    for mode, outcome in outcomes.items():
        print(f"  {mode}: {outcome['images']} images")
        for class_name, (tp, fp, fn) in outcome["counts"].items():
            print(f"    {class_name}: {tp + fn} ground truth, {tp + fp} predictions, {tp} matches, {fp} false positives, {fn} false negatives")
    return outcomes


def main():
    from common import add_common_args, init, read_config_from_file
    from ccipy.omero.omero_getter_ctx import OmeroGetterCtx
    import logging
    from ccipy.utils.cci_logger import CCILogger

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=int, help="Single OMERO dataset ID")
    add_common_args(parser)
    args = parser.parse_args()
    logger = logging.getLogger("omero_evaluation")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    CCILogger._logger = logger
    ids = [args.dataset] if args.dataset is not None else []
    group, host = args.group, args.use_test_host
    if args.config_name:
        common, settings = read_config_from_file("evaluate", args.config_name)
        group = settings.get("omero_group", common.get("omero_group", group))
        host = settings.get("use_test_host", common.get("use_test_host", host))
        ids = settings.get("dataset_ids", [settings["dataset_id"]] if "dataset_id" in settings else ids)
    if not isinstance(ids, list) or not ids or any(type(x) is not int or x <= 0 for x in ids) or len(set(ids)) != len(ids):
        parser.error("Provide a non-empty list of unique positive integer dataset IDs")
    if not group:
        parser.error("Provide an OMERO group via --group or the configuration")
    connection = init(args.token, group, use_test_host=host, init_log=False)
    results = {}
    try:
        with OmeroGetterCtx(connection) as getter:
            for dataset_id in ids:
                print(f"Evaluating dataset {dataset_id}")
                results[dataset_id] = collect_dataset(getter, dataset_id)
    finally:
        connection.close()
    modes = ("all_images", "annotated_images")
    totals = {mode: {name: [sum(results[dataset_id][mode]["counts"][name][i] for dataset_id in ids)
                           for i in range(3)] for name in CLASS_COLORS} for mode in modes}
    vmax = max(1, *(value for mode in modes for counts in totals[mode].values() for value in counts))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y-%m-%d_%H-%M-%S_"), dir=OUTPUT_DIR))
    for mode in modes:
        output_dir = run_dir / mode
        output_dir.mkdir()
        for dataset_id, result in results.items():
            counts = result[mode]["counts"]
            write_csv(output_dir / f"{dataset_id}_evaluation_result.csv", counts)
            for name in CLASS_COLORS:
                plot_matrices(output_dir / f"{dataset_id}_confusion_matrix_{name}.png", counts, vmax, f"Dataset {dataset_id}", [name])
        write_csv(output_dir / "all_datasets.csv", totals[mode])
        for name in CLASS_COLORS:
            plot_matrices(output_dir / f"all_datasets_{name}.png", totals[mode], vmax, "All datasets", [name])
        for case in ("missed", "hallucinated"):
            write_cases(output_dir / f"{case}_cases.csv", [row for dataset_id in ids for row in results[dataset_id][mode][case]])
        (output_dir / "dataset_ids.txt").write_text("".join(f"{dataset_id}\n" for dataset_id in ids))
    print(f"Results saved in {run_dir}/")


if __name__ == "__main__":
    main()
