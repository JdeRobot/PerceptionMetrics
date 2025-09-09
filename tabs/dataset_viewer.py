import streamlit as st
import os
from streamlit_image_select import image_select


def dataset_viewer_tab():
    from detectionmetrics.datasets.coco import CocoDataset
    import supervision as sv
    import numpy as np
    from PIL import Image
    from supervision.draw.color import ColorPalette
    from supervision.detection.annotate import BoxAnnotator
    from supervision.detection.core import Detections

    # Get inputs from session state
    dataset_path = st.session_state.get("dataset_path", "")
    dataset_type = st.session_state.get("dataset_type_selectbox", "Coco")
    split = st.session_state.get("split_selectbox", "val")

    # Header row only
    st.header("Dataset Viewer")

    if not dataset_path or not os.path.isdir(dataset_path):
        st.warning("⚠️ Please select a valid dataset folder.")
        return

    # Setup paths and pagination
    img_dir = os.path.join(
        dataset_path, f"images/{split}2017" if dataset_type.lower() == "coco" else split
    )
    ann_file = os.path.join(
        dataset_path,
        "annotations",
        (
            f"instances_{split}2017.json"
            if dataset_type.lower() == "coco"
            else f"{split}.json"
        ),
    )

    if not os.path.isdir(img_dir) or not os.path.isfile(ann_file):
        st.warning("Dataset files not found. Check path and split.")
        return

    # Pagination and search row
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 2, 1.5])
    with nav_col1:
        pass  # Placeholder for "< page" button, to be added later in the code
    with nav_col2:
        pass  # Placeholder for ">" button, to be added later in the code
    with nav_col3:
        pass  # Placeholder for page info, to be added later in the code
    with nav_col4:
        # Move the button up by reducing the margin and decrease button size with custom CSS
        st.markdown(
            """
            <style>
            div[data-testid="stButton"] button#search_icon_btn {
                padding: 0.15rem 0.5rem;
                font-size: 0.85rem;
                min-height: 1.5rem;
                height: 1.5rem;
                line-height: 1.1;
            }
            /* Move the button up by adding negative top margin */
            div[data-testid="stButton"] {
                margin-top: -0.85rem !important;
            }
            </style>
        """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-bottom: 0;'></div>", unsafe_allow_html=True)

    # Load dataset
    dataset_key = f"{dataset_path}_{split}"
    if dataset_key not in st.session_state:
        try:
            st.session_state[dataset_key] = CocoDataset(
                annotation_file=ann_file,
                image_dir=img_dir,
                split=split,
            )
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            return
    else:
        # Ensure cached dataset has the correct split; if not, rebuild it
        try:
            cached_ds = st.session_state[dataset_key]
            cached_split = getattr(cached_ds, "split", None)
            if cached_split != split:
                st.session_state[dataset_key] = CocoDataset(
                    annotation_file=ann_file,
                    image_dir=img_dir,
                    split=split,
                )
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

    # Pagination
    IMAGES_PER_PAGE = 12
    total_images, total_pages = (
        len(image_files),
        (len(image_files) + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE,
    )
    page_key = f"image_page_{dataset_path}_{split}"

    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    current_page = max(0, min(st.session_state[page_key], total_pages - 1))
    st.session_state[page_key] = current_page

    start_idx = current_page * IMAGES_PER_PAGE
    sample_images = image_files[start_idx : start_idx + IMAGES_PER_PAGE]
    image_paths = [os.path.join(img_dir, img_name) for img_name in sample_images]

    # CSS for compact image grid
    st.markdown(
        """
        <style>
        .image-selector__image, .image-selector__image img {
            max-width: 40px; max-height: 40px; width: 40px; height: 40px; object-fit: contain;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Navigation
    col1, col2, col3, col4 = st.columns([0.5, 9.5, 0.5, 0.5])
    with col1:
        if st.button("⟨", key="prev_page_btn", disabled=(current_page == 0)):
            st.session_state[page_key] = current_page - 1
            st.rerun()
    with col2:
        st.markdown(
            f"<div style='text-align:center;font-weight:bold;'>Page {current_page+1} of {total_pages}</div>",
            unsafe_allow_html=True,
        )
    with col3:
        if st.button(
            "⟩", key="next_page_btn", disabled=(current_page >= total_pages - 1)
        ):
            st.session_state[page_key] = current_page + 1
            st.rerun()
    with col4:
        if st.button(
            "🔍",
            key="search_icon_btn",
            help="Search for an image by name",
            disabled=not (dataset_path and os.path.isdir(dataset_path)),
        ):
            st.session_state["show_search_dropdown"] = True

    # Search dropdown
    if st.session_state.get("show_search_dropdown", False):
        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            selected_img = st.selectbox(
                "Search image:", options=image_files, key="search_image_selectbox"
            )
        with col2:
            st.markdown(
                "<div style='margin-bottom: 2.4rem;'></div>", unsafe_allow_html=True
            )
            if st.button("Go to image", key="go_to_image_btn"):
                new_page = image_files.index(selected_img) // IMAGES_PER_PAGE
                st.session_state[page_key] = new_page
                st.session_state[
                    f"img_select_all_{dataset_path}_{split}_{new_page}"
                ] = (image_files.index(selected_img) % IMAGES_PER_PAGE)
                st.session_state["show_search_dropdown"] = False
                st.rerun()
        with col3:
            st.markdown(
                "<div style='margin-bottom: 2.4rem;'></div>", unsafe_allow_html=True
            )
            if st.button("Cancel", key="cancel_search_btn"):
                st.session_state["show_search_dropdown"] = False
                st.rerun()

    # Image grid
    img_select_key = f"img_select_all_{dataset_path}_{split}_{current_page}"
    img_select_index = st.session_state.get(img_select_key)
    if img_select_index is None or not isinstance(img_select_index, int):
        img_select_index = 0
    selected_img_path = (
        image_select(
            label="",
            images=image_paths,
            captions=sample_images,
            use_container_width=True,
            key=img_select_key,
            index=img_select_index,
        )
        if image_paths
        else None
    )

    # Display selected image with annotations
    if selected_img_path:
        selected_img_name = os.path.basename(selected_img_path)
        try:
            img = Image.open(selected_img_path).convert("RGB")
            img_np = np.array(img)

            ann_row = dataset.dataset[dataset.dataset["image"] == selected_img_name]
            if not ann_row.empty:
                annotation_id = ann_row.iloc[0]["annotation"]
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
                st.image(annotated_pil, use_container_width=False)
            else:
                st.warning("No annotation found for this image.")
        except Exception as e:
            st.error(f"Error displaying image: {e}")
    else:
        st.info("Select an image to view with annotations.")
