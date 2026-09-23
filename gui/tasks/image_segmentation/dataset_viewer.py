from ast import Load
import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from perceptionmetrics.datasets.cityscapes import CityscapesImageSegmentationDataset
from perceptionmetrics.datasets.nuimages import NuImagesSegmentationDataset
from gui.tasks.utils import render_image_grid


def _overlay_mask(image, label, ontology, opacity):
    """Overlay a segmentation mask on an image.
    :param image: PIL Image object of the original image.
    :type image: PIL.Image
    :param label: 2D numpy array of the segmentation mask.
    :type label: np.ndarray
    :param ontology: Dictionary mapping class names to their properties, including 'idx' and 'rgb'.
    :type ontology: dict
    :param opacity: Float value between 0 and 1 indicating the opacity of the overlay.
    :type opacity: float
    :return: PIL Image object of the image with the overlay applied.
    :rtype: PIL.Image
    """
    image_np = np.array(image)
    color_mask = np.zeros((*label.shape, 3), dtype=np.uint8)
    for class_data in ontology.values():
        class_idx = int(class_data["idx"])
        rgb = class_data.get("rgb")
        if rgb is None:
            rng = np.random.default_rng(abs(class_idx))
            rgb = tuple(int(value) for value in rng.integers(0, 255, size=3))
        color_mask[label == class_idx] = rgb

    resampling = getattr(Image, "Resampling", Image)
    color_mask_image = Image.fromarray(color_mask).resize(
        image.size, resampling.NEAREST
    )
    color_mask_np = np.array(color_mask_image)

    overlay = ((1.0 - opacity) * image_np + opacity * color_mask_np).astype(np.uint8)
    return Image.fromarray(overlay)


def render_image_segmentation_viewer():
    """Load the selected image segmentation dataset and render the viewer tab."""
    dataset_type = st.session_state.get("segmentation_dataset_type", "Cityscapes")
    dataset_path = st.session_state.get("dataset_path", "")
    split = st.session_state.get("split", "val")

    st.header("Dataset Viewer")

    if not dataset_path or not os.path.isdir(dataset_path):
        st.warning("Please select a valid image segmentation dataset folder.")
        return

    try:
        dataset = load_image_segmentation_dataset(dataset_type, dataset_path, split)
    except Exception as exc:
        st.error(f"Failed to load image segmentation dataset: {exc}")
        return

    display_loaded_segmentation_dataset(
        dataset=dataset,
        dataset_type=dataset_type,
        split=split,
        state_prefix="image_segmentation",
        context=f"{dataset_path}_{split}",
    )


def display_loaded_segmentation_dataset(
    dataset,
    dataset_type,
    split,
    state_prefix,
    context,
):
    """
    Display samples, masks, and class information for a loaded dataset.
    :param dataset: Instance of the loaded image segmentation dataset.
    :type dataset : CityscapesImageSegmentationDataset or NuImagesSegmentationDataset
    :param dataset_type: Type of the dataset (e.g., "Cityscapes", "NuImages")
    :type dataset_type : str
    :param split: Dataset split to display (e.g., "train", "val", "test")
    :type split : str
    :param state_prefix: Prefix for Streamlit session state keys to avoid collisions.
    :type state_prefix : str
    :param context: Context string for the dataset, used in session state keys.
    :type context : str
    """
    split_df = dataset.dataset[dataset.dataset["split"] == split]
    if split_df.empty:
        st.warning(f"No {dataset_type} samples found for split '{split}'.")
        return

    sample_names = split_df.index.astype(str).tolist()
    image_paths = split_df["image"].tolist()
    selected_img_path, sample_name = render_image_grid(
        item_names=sample_names,
        image_paths=image_paths,
        state_prefix=state_prefix,
        context=context,
        search_label="sample",
    )

    if not selected_img_path:
        st.info("Select an image to view the ground truth mask.")
        return

    mask_opacity = st.slider(
        "Mask Opacity",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        key=f"{state_prefix}_mask_opacity",
    )

    row_key = sample_name
    if row_key not in split_df.index and sample_name.isdigit():
        row_key = int(sample_name)

    row = split_df.loc[row_key]
    image_fname = row["image"]
    label_fname = row["label"]

    try:
        image = Image.open(image_fname).convert("RGB")
        label = dataset.read_label(label_fname)
    except Exception as exc:
        st.error(f"Failed to read sample '{sample_name}': {exc}")
        return

    overlay = _overlay_mask(image, label, dataset.ontology, mask_opacity)

    image_col, overlay_col = st.columns(2)
    with image_col:
        st.image(image, caption="Image", use_container_width=True)
    with overlay_col:
        st.image(overlay, caption="Ground Truth Overlay", use_container_width=True)

    with st.expander("Classes", expanded=False):
        st.dataframe(_classes_dataframe(dataset.ontology), use_container_width=True)


def load_image_segmentation_dataset(dataset_type, dataset_path, split):
    """Load an image segmentation dataset based on the provided type, path, and split.
    :param dataset_type: Type of the dataset (e.g., "Cityscapes", "NuImages")
    :type dataset_type : str
    :param dataset_path: Path to the dataset directory.
    :type dataset_path : str
    :param split: Dataset split to load (e.g., "train", "val", "test")
    :type split : str
    :return: Instance of the loaded image segmentation dataset.
    :rtype: CityscapesImageSegmentationDataset or NuImagesSegmentationDataset
    """
    if dataset_type == "Cityscapes":
        return load_cityscapes_dataset(dataset_path, split)
    if dataset_type == "NuImages":
        return load_nuimages_dataset(dataset_path, split)
    raise ValueError(f"{dataset_type} image segmentation dataset is not wired yet.")


def load_cityscapes_dataset(dataset_path, split):
    """
    Load the Cityscapes dataset based on the provided path and split.
    :param dataset_path: Path to the Cityscapes dataset directory.
    :type dataset_path: str
    :param split: Dataset split to load (e.g., "train", "val", "test").
    :type split: str
    :return: Instance of CityscapesImageSegmentationDataset as a session state variable.
    :rtype: CityscapesImageSegmentationDataset
    """
    roots = {"train": None, "val": None, "test": None}
    roots[split] = dataset_path

    dataset_key = (
        "cityscapes_segmentation_dataset",
        os.path.abspath(dataset_path),
        split,
        st.session_state.get(
            "segmentation_image_dir", "leftImg8bit_trainvaltest/leftImg8bit"
        ),
        st.session_state.get("segmentation_label_dir", "gtFine"),
        st.session_state.get("segmentation_image_suffix", "_leftImg8bit.png"),
        st.session_state.get("segmentation_label_suffix", "_gtFine_labelIds.png"),
        st.session_state.get("segmentation_use_train_id", False),
    )

    if dataset_key not in st.session_state:
        st.session_state[dataset_key] = CityscapesImageSegmentationDataset(
            train_dataset_root=roots["train"],
            val_dataset_root=roots["val"],
            test_dataset_root=roots["test"],
            image_dir=dataset_key[3],
            label_dir=dataset_key[4],
            image_suffix=dataset_key[5],
            label_suffix=dataset_key[6],
            use_train_id=dataset_key[7],
        )

    return st.session_state[dataset_key]


def load_nuimages_dataset(dataset_path, split):
    """Load the NuImages dataset based on the provided path and split.
    :param dataset_path: Path to the NuImages dataset directory.
    :type dataset_path: str
    :param split: Dataset split to load (e.g., "train", "val").
    :type split: str
    :return: Instance of NuImagesSegmentationDataset as a session state variable.
    :rtype: NuImagesSegmentationDataset
    """
    dataset_key = (
        "nuimages_segmentation_dataset",
        os.path.abspath(dataset_path),
        split,
        st.session_state.get("nuimages_segmentation_version", "v1.0-mini"),
        st.session_state.get(
            "nuimages_segmentation_labels_dir",
            "generated/nuimages_segmentation_labels",
        ),
    )

    if dataset_key not in st.session_state:
        st.session_state[dataset_key] = NuImagesSegmentationDataset(
            dataset_dir=dataset_path,
            version=dataset_key[3],
            split=dataset_key[2],
            labels_rel_dir=dataset_key[4],
        )

    return st.session_state[dataset_key]


def _classes_dataframe(ontology):
    """Convert the ontology dictionary to a pandas DataFrame for display.
    :param ontology: Dictionary mapping class names to their properties, including 'idx', 'train_id', 'category', and 'rgb'.
    :type ontology: dict
    :return: pandas DataFrame with columns ['class', 'id', 'train_id', 'category', 'rgb'].
    :rtype: pd.DataFrame
    """
    rows = []
    for class_name, class_data in ontology.items():
        rows.append(
            {
                "class": class_name,
                "id": class_data["idx"],
                "train_id": class_data.get("train_id"),
                "category": class_data.get("category"),
                "rgb": class_data.get("rgb"),
            }
        )
    return pd.DataFrame(rows)
