import streamlit as st
import os
import tempfile

from perceptionmetrics.datasets.coco import find_img_dir_and_ann_file
from gui.tasks.utils import render_image_grid


def render_image_detection_viewer():
    """Render the image detection dataset viewer tab in Streamlit."""
    import numpy as np
    from PIL import Image
    from perceptionmetrics.utils.image import draw_detections

    # Get inputs from session state
    dataset_path = st.session_state.get("dataset_path", "")
    dataset_type = st.session_state.get("dataset_type", "COCO").lower()
    split = st.session_state.get("split", "val")

    # Header row only
    st.header("Dataset Viewer")

    if not dataset_path or not os.path.isdir(dataset_path):
        st.warning("⚠️ Please select a valid dataset folder.")
        return

    # Setup image directory used for pagination
    if dataset_type == "coco":
        try:
            img_dir, _ = find_img_dir_and_ann_file(
                dataset_path=dataset_path, split=split
            )
        except FileNotFoundError:
            st.warning("Dataset files not found. Check path and split.")
            return
    elif dataset_type == "yolo":
        img_dir = os.path.join(dataset_path, "images", split)
        if not os.path.isdir(img_dir):
            st.warning("Image directory not found.")
            return
    else:
        st.error("Unsupported dataset type.")
        return

    # Load dataset (shared with the evaluator tab through session state)
    dataset_key = get_image_detection_dataset_key(dataset_type, dataset_path, split)
    if dataset_key not in st.session_state:
        try:
            st.session_state[dataset_key] = load_image_detection_dataset(
                dataset_type,
                dataset_path,
                split,
                st.session_state.get("dataset_config_file"),
            )
        except FileNotFoundError as e:
            st.warning(f"⚠️ {e}")
            return
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            return
    dataset = st.session_state[dataset_key]

    # Get image files
    image_files = [
        f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not image_files:
        st.warning("No images found.")
        return

    image_paths = [os.path.join(img_dir, img_name) for img_name in image_files]
    selected_img_path, _ = render_image_grid(
        item_names=image_files,
        image_paths=image_paths,
        state_prefix="image_detection",
        context=f"{dataset_path}_{split}",
        search_label="image",
    )

    # Display selected image with annotations
    if selected_img_path:
        selected_img_name = os.path.basename(selected_img_path)
        try:
            img = Image.open(selected_img_path).convert("RGB")

            if dataset_type == "yolo":
                ann_row = dataset.dataset[
                    dataset.dataset["image"].str.endswith(selected_img_name)
                ]
            else:
                ann_row = dataset.dataset[dataset.dataset["image"] == selected_img_name]

            if not ann_row.empty:
                annotation_id = ann_row.iloc[0]["annotation"]
                if dataset_type == "yolo":
                    annotation_id = os.path.join(dataset.dataset_dir, annotation_id)

                boxes, category_indices = dataset.read_annotation(annotation_id)

                # Get class names from ontology
                ontology = getattr(dataset, "ontology", None)
                if ontology is None and hasattr(dataset.dataset, "attrs"):
                    ontology = dataset.dataset.attrs.get("ontology", None)

                if ontology:
                    catid_to_name = {v["idx"]: k for k, v in ontology.items()}
                    class_names = [
                        catid_to_name.get(cat_id, str(cat_id))
                        for cat_id in category_indices
                    ]
                else:
                    class_names = [str(cat_id) for cat_id in category_indices]

                # Annotate image using the shared, version-adaptive helper
                annotated_img = draw_detections(
                    image=img,
                    boxes=np.array(boxes),
                    class_ids=np.array(category_indices),
                    class_names=class_names,
                )

                # Resize for display
                annotated_pil = Image.fromarray(annotated_img)
                try:
                    resample = getattr(Image, "Resampling", Image).LANCZOS
                except AttributeError:
                    resample = Image.LANCZOS
                annotated_pil.thumbnail((500, 500), resample)
                st.image(annotated_pil, width="content")
            else:
                st.warning("No annotation found for this image.")
        except Exception as e:
            st.error(f"Error displaying image: {e}")
    else:
        st.info("Select an image to view with annotations.")


def get_image_detection_dataset_key(dataset_type, dataset_path, split):
    """Build the session state key used to cache an image detection dataset.

    The dataset type is part of the key so that switching between COCO and YOLO
    for the same folder and split never returns a stale dataset.

    :param dataset_type: Type of the dataset (e.g., "COCO", "YOLO")
    :type dataset_type: str
    :param dataset_path: Path to the dataset directory
    :type dataset_path: str
    :param split: Dataset split (e.g., "train", "val", "test")
    :type split: str
    :return: Session state key for the dataset
    :rtype: str
    """
    return f"{dataset_type.lower()}_{dataset_path}_{split}"


def load_image_detection_dataset(
    dataset_type, dataset_path, split, dataset_config_file=None
):
    """Load an image detection dataset based on the provided type, path, and split.

    Filenames are kept relative to the dataset directory. Models make them global
    on their own copy of the dataset during evaluation.

    :param dataset_type: Type of the dataset ("COCO" or "YOLO", case-insensitive)
    :type dataset_type: str
    :param dataset_path: Path to the dataset directory
    :type dataset_path: str
    :param split: Dataset split to load (e.g., "train", "val", "test")
    :type split: str
    :param dataset_config_file: YOLO dataset configuration (YAML) as an uploaded
        file-like object exposing ``getvalue()``. Required for YOLO datasets.
    :type dataset_config_file: Optional[streamlit.runtime.uploaded_file_manager.UploadedFile]
    :raises FileNotFoundError: If required dataset files or configuration are missing
    :raises ValueError: If the dataset type is not supported
    :return: Loaded dataset containing only the requested split
    :rtype: CocoDataset or YOLODataset
    """
    from perceptionmetrics.datasets.coco import CocoDataset
    from perceptionmetrics.datasets.yolo import YOLODataset

    dataset_type = dataset_type.lower()

    if dataset_type == "coco":
        img_dir, ann_file = find_img_dir_and_ann_file(
            dataset_path=dataset_path, split=split
        )
        return CocoDataset(annotation_file=ann_file, image_dir=img_dir, split=split)

    if dataset_type == "yolo":
        if dataset_config_file is None:
            raise FileNotFoundError(
                "Please upload the YOLO dataset configuration (.yaml) in the sidebar."
            )

        # getvalue() returns the full content even if the file was read before
        with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as tmp:
            tmp.write(dataset_config_file.getvalue())
            tmp_path = tmp.name
        try:
            yolo_dataset = YOLODataset(tmp_path, dataset_path)
        finally:
            os.unlink(tmp_path)

        yolo_dataset.dataset = yolo_dataset.dataset[
            yolo_dataset.dataset["split"] == split
        ].reset_index(drop=True)
        return yolo_dataset

    raise ValueError(f"{dataset_type} image detection dataset is not supported.")
