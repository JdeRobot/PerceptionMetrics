import streamlit as st
import os

from perceptionmetrics.datasets.coco import find_img_dir_and_ann_file
from gui.tasks.utils import render_image_grid


def render_image_detection_viewer():
    """Render the image detection dataset viewer tab in Streamlit."""
    import tempfile
    from perceptionmetrics.datasets.coco import CocoDataset
    from perceptionmetrics.datasets.yolo import YOLODataset
    import numpy as np
    from PIL import Image
    from supervision.draw.color import ColorPalette
    from supervision.detection.annotate import BoxAnnotator
    from supervision.detection.core import Detections

    # Get inputs from session state
    dataset_path = st.session_state.get("dataset_path", "")
    dataset_type = st.session_state.get("dataset_type", "COCO").lower()
    split = st.session_state.get("split", "val")

    # Header row only
    st.header("Dataset Viewer")

    if not dataset_path or not os.path.isdir(dataset_path):
        st.warning("⚠️ Please select a valid dataset folder.")
        return

    # Setup paths and pagination
    if dataset_type == "coco":
        try:
            img_dir, ann_file = find_img_dir_and_ann_file(
                dataset_path=dataset_path, split=split
            )
        except FileNotFoundError:
            st.warning("Dataset files not found. Check path and split.")
            return

    elif dataset_type == "yolo":
        dataset_config_file = st.session_state.get("dataset_config_file", None)
        img_dir = os.path.join(dataset_path, f"images/{split}")
        if not os.path.isdir(img_dir):
            st.warning("Image directory not found.")
            return
        if dataset_config_file is None:
            st.warning("Dataset configuration file not found. Please upload it.")
            return
    else:
        st.error("Unsupported dataset type.")
        return

    # Load dataset
    dataset_key = f"{dataset_path}_{split}"
    if dataset_key not in st.session_state:
        try:
            if dataset_type == "coco":
                st.session_state[dataset_key] = CocoDataset(
                    annotation_file=ann_file,
                    image_dir=img_dir,
                    split=split,
                )
            elif dataset_type == "yolo":
                if dataset_config_file is not None:
                    # Save uploaded config file to a temporary location
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".yaml"
                    ) as tmp:
                        tmp.write(dataset_config_file.read())
                        tmp_path = tmp.name

                    # Load YOLO dataset
                    yolo_dataset = YOLODataset(tmp_path, dataset_path)
                    st.session_state["full_dataset_df"] = yolo_dataset.dataset

                    # Filter dataset for the selected split
                    yolo_dataset.dataset = yolo_dataset.dataset[
                        yolo_dataset.dataset["split"] == split
                    ].reset_index(drop=True)
                    st.session_state[dataset_key] = yolo_dataset

                    os.unlink(tmp_path)  # Clean up temp file
                else:
                    st.warning(
                        "Dataset configuration file not found. Please upload it."
                    )
                    return
            else:
                st.error("Unsupported dataset type.")
                return

        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            return
    else:
        # Ensure cached dataset has the correct split; if not, rebuild it
        try:
            cached_ds = st.session_state[dataset_key]
            cached_split = getattr(cached_ds, "split", None)
            if cached_split != split:
                if dataset_type == "coco":
                    st.session_state[dataset_key] = CocoDataset(
                        annotation_file=ann_file,
                        image_dir=img_dir,
                        split=split,
                    )
                elif dataset_type == "yolo":
                    yolo_dataset = st.session_state[dataset_key]
                    yolo_dataset.dataset = st.session_state["full_dataset_df"][
                        st.session_state["full_dataset_df"]["split"] == split
                    ].reset_index(drop=True)
                    st.session_state[dataset_key] = yolo_dataset
                else:
                    st.error("Unsupported dataset type.")
                    return
        except Exception:
            pass
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
            img_np = np.array(img)

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

                # Annotate image
                palette = ColorPalette.default()
                detections = Detections(
                    xyxy=np.array(boxes), class_id=np.array(category_indices)
                )
                annotator = BoxAnnotator(
                    color=palette, text_scale=0.7, text_thickness=1, text_padding=2
                )
                annotated_img = annotator.annotate(
                    scene=img_np, detections=detections, labels=class_names
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
