"""
Convert the QuPath polygon annotations (geojson files) to masks with labels for the two types of rare granules
"""

import json
import numpy as np
from skimage.draw import polygon
from skimage.io import imsave
from pathlib import Path


def geojson_to_labels(geojson_path, image_shape):
    class_to_label = {
        "granule": 1,
        "tail": 2,
        "spike": 3,
    }

    labels = np.zeros(image_shape, dtype=np.uint8)

    with open(geojson_path, "r") as f:
        data = json.load(f)

    for feature in data["features"]:
        geometry = feature["geometry"]
        properties = feature.get("properties", {})

        # QuPath commonly stores the class like:
        # properties["classification"]["name"]
        classification = properties.get("classification")

        if classification is None:
            continue

        class_name = classification.get("name")

        if class_name not in class_to_label:
            continue

        label_value = class_to_label[class_name]

        if geometry["type"] == "Polygon":
            polygons = [geometry["coordinates"]]

        elif geometry["type"] == "MultiPolygon":
            polygons = geometry["coordinates"]

        else:
            continue

        for poly in polygons:
            # Exterior polygon
            coords = np.asarray(poly[0])

            x = coords[:, 0]
            y = coords[:, 1]

            rr, cc = polygon(y, x, shape=image_shape)
            labels[rr, cc] = label_value

            # Polygon holes should remain background
            for hole in poly[1:]:
                hole = np.asarray(hole)

                x_hole = hole[:, 0]
                y_hole = hole[:, 1]

                rr, cc = polygon(y_hole, x_hole, shape=image_shape)
                labels[rr, cc] = 0

    return labels


geojson_dir = Path("/home/xgupke/Documents/projects/EmmaElinTrombocytes/datafiles/images/annotations_rare_granules")
data_root_dir = geojson_dir.parent
labels_dir = data_root_dir / "labels"
labels_dir.mkdir(exist_ok=True)


names_to_exclude = ["40135_J5_new_x4300_-10um_0002.geojson",
                    "40135_G1_x4300_-10um_0095.geojson"]

for gfile in geojson_dir.glob("*.geojson"):
    if gfile.name not in names_to_exclude:
        continue

    print(gfile)
    labels = geojson_to_labels(gfile, (4096, 4096))

    labels_save_path = labels_dir / f"{gfile.stem}.png"
    imsave(labels_save_path, labels.astype(np.uint8))

    print(f"    Labels saved to {labels_save_path}")

