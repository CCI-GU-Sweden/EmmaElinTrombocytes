from pathlib import Path

import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.measure import label, regionprops

from scipy.ndimage import gaussian_filter, gaussian_laplace
from skimage.filters import sobel


def get_region_characteristics(labels_dir):

    sizes_dict = {}
    axes_dict = {}

    for f in list(labels_dir.glob("*.png")):
        file = f.name
        labels = imread(f"{labels_dir}/{file}")

        for class_value in np.unique(labels):
            if class_value == 0:
                continue

            binary = labels == class_value
            connected = label(binary)

            for region in regionprops(connected):

                # find the area of the regions
                sizes_dict.setdefault(class_value, []).append(region.area)

                # find the major and minor axes
                axes_dict.setdefault(
                    class_value,
                    {"major": [], "minor": []}
                )
        
                axes_dict[class_value]["major"].append(region.axis_major_length)
                axes_dict[class_value]["minor"].append(region.axis_minor_length)

    return sizes_dict, axes_dict


# Function to go through all the filenames, extract their cell bounds in a list, then make crops from the image and labels.
# After that the pixel feature extractor will be run on the crops

def crop_extractor(filenames, images_dir, labels_dir, csv_name):

    df = pd.read_csv(images_dir / csv_name)
    columns = ['x topleft pixels', 'y topleft pixels', 'width pixels', 'height pixels']

    crops_dict = {}

    for file in filenames:

        print(file)
        img_crops = []
        label_crops = []
        img = imread(images_dir / file)
        labels = imread(labels_dir / file)

        df_cells = df[
            df["image name"].str.replace(r"\.ome\.tiff?$", "", regex=True) == file.removesuffix(".png")
        ]

        df_cells = df_cells.drop_duplicates(subset=columns, keep='last')
        df_cells = df_cells[df_cells['class name'] == 'cell']

        for _, row in df_cells.iterrows():
            x_tl = round(row["x topleft pixels"])
            y_tl = round(row["y topleft pixels"])
            w = round(row["width pixels"])
            h = round(row["height pixels"])

            if h > 500 and w > 500:
                img_crops.append(img[y_tl:y_tl + h, x_tl:x_tl + w])
                label_crops.append(labels[y_tl:y_tl + h, x_tl:x_tl + w])

        crops_dict[file] = {'img_crops': img_crops, 'label_crops': label_crops}

        del img, labels

    return crops_dict


from scipy.ndimage import gaussian_filter, sobel, gaussian_laplace
import numpy as np


def extract_pixel_features(
    image,
    labels,
    stats,
    sigma_fractions=(0.10, 0.20, 0.35),
    truncate=4.0,
):
    """
    Extract multiscale features for pixel classification.

    Features:
        - Raw intensity
        - Gaussian-smoothed intensity
        - Sobel edge magnitude
        - Laplacian of Gaussian
        - Hessian eigenvalues
        - Difference of Gaussian (DoG)

    """

    image = np.asarray(image, dtype=np.float32)
    labels = np.asarray(labels)

    if image.ndim != 2:
        raise ValueError("image must be a 2D grayscale image")

    if labels.shape != image.shape:
        raise ValueError("image and labels must have the same shape")


    # ------------------------------------------------------------
    # Determine characteristic object width
    # ------------------------------------------------------------

    minor_means = np.array([
        class_stats["minor"]["mean"]
        for class_stats in stats.values()
    ])

    characteristic_width = np.median(minor_means)


    # ------------------------------------------------------------
    # Choose sigma values
    # ------------------------------------------------------------

    sigmas = characteristic_width * np.asarray(sigma_fractions)

    kernel_sizes = (
        2 * np.ceil(truncate * sigmas).astype(int) + 1
    )


    # ------------------------------------------------------------
    # Build feature map
    # ------------------------------------------------------------

    features = []
    feature_names = []

    # Raw intensity
    features.append(image)
    feature_names.append("intensity")


    # Store Gaussian images because DoG needs them
    gaussian_images = {}


    for sigma in sigmas:

        # ---------- Gaussian ----------
        gaussian = gaussian_filter(
            image,
            sigma=sigma,
            truncate=truncate,
        )

        gaussian_images[sigma] = gaussian

        features.append(gaussian)
        feature_names.append(
            f"gaussian_sigma_{sigma:.2f}"
        )


        # ---------- Sobel ----------
        sobel_feature = sobel(gaussian)

        features.append(sobel_feature)
        feature_names.append(
            f"sobel_sigma_{sigma:.2f}"
        )


        # ---------- LoG ----------
        log_feature = gaussian_laplace(
            image,
            sigma=sigma,
            truncate=truncate,
        )

        log_feature *= sigma ** 2

        features.append(log_feature)
        feature_names.append(
            f"LoG_sigma_{sigma:.2f}"
        )


        # ---------- Hessian ----------
        Hyy = gaussian_filter(
            image,
            sigma=sigma,
            order=(2, 0),
            truncate=truncate,
        )

        Hxy = gaussian_filter(
            image,
            sigma=sigma,
            order=(1, 1),
            truncate=truncate,
        )

        Hxx = gaussian_filter(
            image,
            sigma=sigma,
            order=(0, 2),
            truncate=truncate,
        )

        Hxx *= sigma ** 2
        Hxy *= sigma ** 2
        Hyy *= sigma ** 2


        trace = Hxx + Hyy
        difference = Hxx - Hyy

        root = np.sqrt(
            difference ** 2 +
            4 * Hxy ** 2
        )

        eig1 = (trace + root) / 2
        eig2 = (trace - root) / 2

        features.append(eig1)
        feature_names.append(
            f"hessian_eig1_sigma_{sigma:.2f}"
        )

        features.append(eig2)
        feature_names.append(
            f"hessian_eig2_sigma_{sigma:.2f}"
        )


    # ------------------------------------------------------------
    # Difference of Gaussian features
    # ------------------------------------------------------------

    # Adjacent scales + largest scale difference
    dog_pairs = [
        (sigmas[0], sigmas[1]),
        (sigmas[1], sigmas[2]),
        (sigmas[0], sigmas[2]),
    ]

    for sigma1, sigma2 in dog_pairs:

        dog = (
            gaussian_images[sigma1]
            -
            gaussian_images[sigma2]
        )

        features.append(dog)

        feature_names.append(
            f"DoG_sigma_{sigma1:.2f}_{sigma2:.2f}"
        )


    # ------------------------------------------------------------
    # Stack into feature map
    # ------------------------------------------------------------

    feature_map = np.stack(
        features,
        axis=-1
    )


    # ------------------------------------------------------------
    # Convert pixels into rows
    # ------------------------------------------------------------

    n_features = feature_map.shape[-1]

    X = feature_map.reshape(-1, n_features)
    y = labels.reshape(-1, 1)

    data = np.concatenate(
        (X, y),
        axis=1
    )

    return feature_map, data, feature_names, sigmas, kernel_sizes