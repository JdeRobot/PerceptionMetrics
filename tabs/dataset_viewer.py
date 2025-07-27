import streamlit as st
import os
import sys
import subprocess
from streamlit_image_select import image_select

def browse_folder():
    """
    Opens a native folder selection dialog and returns the selected folder path.
    Works on Windows, macOS, and Linux (with zenity or kdialog).
    Returns None if cancelled or error.
    """
    try:
        if sys.platform.startswith("win"):
            script = (
                'Add-Type -AssemblyName System.windows.forms;'
                '$f=New-Object System.Windows.Forms.FolderBrowserDialog;'
                'if($f.ShowDialog() -eq "OK"){Write-Output $f.SelectedPath}'
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True, text=True, timeout=30
            )
            folder = result.stdout.strip()
            return folder if folder else None
        elif sys.platform == "darwin":
            script = 'POSIX path of (choose folder with prompt "Select dataset folder:")'
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=30
            )
            folder = result.stdout.strip()
            return folder if folder else None
        else:
            # Linux: try zenity, then kdialog
            for cmd in [
                ["zenity", "--file-selection", "--directory", "--title=Select dataset folder"],
                ["kdialog", "--getexistingdirectory", "--title", "Select dataset folder"]
            ]:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    folder = result.stdout.strip()
                    if folder:
                        return folder
                except Exception:
                    continue
            return None
    except Exception:
        return None

def dataset_viewer_tab():
    from detectionmetrics.datasets.coco import CocoDataset

    st.header("Dataset Viewer")
    # Ensure dataset_path is initialized in session state before use
    st.session_state.setdefault("dataset_path", "")
    # --- Input selectors in a single row ---
    col1, col2, col3, col4, col5 = st.columns([1, 4, 1, 1, 0.7])

    with col1:
        dataset_type = st.selectbox("Type", ["Coco", "Custom"], key="dataset_type_selectbox")
    with col2:
        dataset_path = st.text_input(
            "Dataset Folder Path",
            value=st.session_state["dataset_path"],
            key="dataset_path_input"
        )
    with col3:
        st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
        if st.button("Browse", key="browse_button"):
            folder = browse_folder()
            if folder and os.path.isdir(folder):
                st.session_state["dataset_path"] = folder
                st.rerun()
            elif folder is not None:
                st.warning("Selected path is not a valid folder.")
    with col4:
        split_disabled = not (dataset_path and os.path.isdir(dataset_path))
        split = st.selectbox(
            "Split", ["train", "val"], key="split_selectbox", disabled=split_disabled
        )
    with col5:
        st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
        if st.button("🔍", key="search_icon_btn", help="Search for an image by name", disabled=split_disabled):
            st.session_state["show_search_dropdown"] = True

    # Sync session state with text input
    if dataset_path != st.session_state["dataset_path"]:
        st.session_state["dataset_path"] = dataset_path
    dataset_path = st.session_state["dataset_path"]

    if not dataset_path:
        return

    if not os.path.isdir(dataset_path):
        st.error("Invalid folder path.")
        return

    # Reset page and rerun when split changes ---
    split_key = "prev_split"
    page_key = f"image_page_{dataset_path}_{split}"
    
    # Initialize or reset page state when split changes
    if split_key not in st.session_state:
        st.session_state[split_key] = split
        st.session_state[page_key] = 0
    elif st.session_state[split_key] != split:
        # Clear the old page key and set the new one
        old_page_key = f"image_page_{dataset_path}_{st.session_state[split_key]}"
        if old_page_key in st.session_state:
            del st.session_state[old_page_key]
        st.session_state[page_key] = 0
        st.session_state[split_key] = split
        st.rerun()

    # Assign img_dir and ann_file based on split and dataset_type
    if dataset_type.lower() == "coco":
        img_dir = os.path.join(dataset_path, f"images/{split}2017")
        ann_file = os.path.join(dataset_path, "annotations", f"instances_{split}2017.json")
    else:
        img_dir = os.path.join(dataset_path, split)
        ann_file = os.path.join(dataset_path, "annotations", f"{split}.json")

    # Instantiate dataset class after getting img_dir and ann_file
    dataset = None
    dataset_key = f"{dataset_path}_{split}"
    if dataset_type.lower() == "coco":
        if os.path.isdir(img_dir) and os.path.isfile(ann_file):
            try:
                if dataset_key not in st.session_state:
                    st.session_state[dataset_key] = CocoDataset(annotation_file=ann_file, image_dir=img_dir)
                dataset = st.session_state[dataset_key]
            except Exception as e:
                st.error(f"Failed to load COCO dataset: {e}")
                return
        else:
            if not os.path.isdir(img_dir):
                st.warning("Image directory does not exist." )
            elif not os.path.isfile(ann_file):
                st.write(ann_file)
                st.warning("Annotation file does not exist.")
            return
    else:
        if not os.path.isdir(img_dir):
            st.warning("Image directory does not exist.")
            return
        if not os.path.isfile(ann_file):
            st.warning("Annotation file does not exist.")
            return
        # Placeholder for custom dataset class instantiation
        # dataset = CustomDataset(annotation_file=ann_file, image_dir=img_dir)

    if dataset is None:
        # If dataset instantiation failed or not implemented, stop here
        return

    # Use dataset.dataset (the DataFrame) to get image file names
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        st.warning("No images found in the selected folder.")
        return

    # --- Begin Pagination Logic ---
    IMAGES_PER_PAGE = 12
    total_images = len(image_files)
    total_pages = (total_images + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE

    # Use a unique key for session state based on dataset_path and split
    # page_key = f"image_page_{dataset_path}_{split}" # Moved outside the if block
    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    current_page = st.session_state[page_key]

    # Clamp current_page to valid range
    if current_page < 0:
        current_page = 0
        st.session_state[page_key] = 0
    if current_page > total_pages - 1:
        current_page = total_pages - 1
        st.session_state[page_key] = total_pages - 1

    start_idx = current_page * IMAGES_PER_PAGE
    end_idx = min(start_idx + IMAGES_PER_PAGE, total_images)
    sample_images = image_files[start_idx:end_idx]
    image_paths = [os.path.join(img_dir, img_name) for img_name in sample_images]
    # --- End Pagination Logic ---

    # Inject CSS to make images in the grid smaller (fallback for image_select)
    st.markdown(
        """
        <style>
        .image-selector__image, .image-selector__image img {
            max-width: 40px ;
            max-height: 40px ;
            width:40px ;
            height: 40px ;
            object-fit: contain ;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Show dropdown if search is active (now above the navigation row)
    if st.session_state.get("show_search_dropdown", False):
        st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
        search_col1, search_col2, search_col3 = st.columns([4, 1, 1])
        with search_col1:
            selected_img_name_search = st.selectbox(
                "Search and select an image:",
                options=image_files,
                key="search_image_selectbox"
            )
        with search_col2:
            st.markdown("<div style='margin-bottom: 1.75rem;'></div>", unsafe_allow_html=True)
            if st.button("Go to image", key="go_to_image_btn"):
                # Find index of selected image
                selected_idx = image_files.index(selected_img_name_search)
                # Calculate page
                new_page = selected_idx // IMAGES_PER_PAGE
                # Find the index of the image in the new page's image_paths
                page_image_files = image_files[new_page * IMAGES_PER_PAGE : min((new_page + 1) * IMAGES_PER_PAGE, total_images)]
                try:
                    idx_in_page = page_image_files.index(selected_img_name_search)
                except ValueError:
                    idx_in_page = 0  # fallback to first image if not found
                st.session_state[page_key] = new_page
                # Force Streamlit to use the index argument by deleting the key
                if f"img_select_all_{new_page}" in st.session_state:
                    del st.session_state[f"img_select_all_{new_page}"]
                st.session_state[f"img_select_all_{new_page}"] = idx_in_page
                st.session_state["show_search_dropdown"] = False
                st.rerun()
        with search_col3:
            st.markdown("<div style='margin-bottom: 1.75rem;'></div>", unsafe_allow_html=True)
            if st.button("Cancel", key="cancel_search_btn"):
                st.session_state["show_search_dropdown"] = False
                st.rerun()

    # Navigation row: <, page number, >
    nav_col1, nav_col2, nav_col3 = st.columns([1, 9.5, 1])
    with nav_col1:
        if st.button("⟨", key="prev_page_btn", disabled=(current_page == 0)):
            st.session_state[page_key] = max(0, current_page - 1)
            st.rerun()
    with nav_col2:
        st.markdown(
            f"<div style='text-align:center;font-weight:bold;'>Select an image (Page {current_page+1} of {total_pages})</div>",
            unsafe_allow_html=True,
        )
    with nav_col3:
        if st.button("⟩", key="next_page_btn", disabled=(current_page >= total_pages - 1)):
            st.session_state[page_key] = min(total_pages - 1, current_page + 1)
            st.rerun()

    # Show all images in the current page in a single image_select, then display below
    # Determine which image to highlight (default to 0)
    img_select_index = st.session_state.get(f"img_select_all_{current_page}")
    if img_select_index is None or not isinstance(img_select_index, int):
        img_select_index = 0
    selected_img_path = image_select(
        label="",
        images=image_paths,
        captions=sample_images,
        use_container_width=True,
        key=f"img_select_all_{current_page}",
        index=img_select_index
    ) if image_paths else None

    if selected_img_path is not None:
        selected_img_name = os.path.basename(selected_img_path)
        try:
            import supervision as sv
            import numpy as np
            from PIL import Image

            img = Image.open(selected_img_path).convert("RGB")
            img_np = np.array(img)

            ann_row = dataset.dataset[dataset.dataset["image"] == selected_img_name]
            if not ann_row.empty:
                annotation_id = ann_row.iloc[0]["annotation"]
                boxes, labels, category_ids = dataset.read_annotation(annotation_id)

                # Get ontology for color coding and class name mapping
                ontology = getattr(dataset, "ontology", None)
                if ontology is None and hasattr(dataset.dataset, "attrs"):
                    ontology = dataset.dataset.attrs.get("ontology", None)

                # Prepare class names and unique class names
                if ontology is not None:
                    catid_to_name = {v["idx"]: k for k, v in ontology.items()}
                    class_names = [catid_to_name.get(cat_id, str(cat_id)) for cat_id in category_ids]
                else:
                    class_names = [str(cat_id) for cat_id in category_ids]
                unique_class_names = list({name for name in class_names})

                from supervision.draw.color import ColorPalette
                from supervision.detection.annotate import BoxAnnotator
                from supervision.detection.core import Detections

                palette = ColorPalette.default()
                class_name_to_color = {name: palette.by_idx(i) for i, name in enumerate(unique_class_names)}
                box_colors = [class_name_to_color[name] for name in class_names]

                # Prepare detections for supervision
                xyxy = np.array(boxes)
                class_id = np.array(category_ids)  # Use integer category IDs

                detections = Detections(
                    xyxy=xyxy,
                    class_id=class_id
                )
                # Annotate with class names (not just IDs) using the labels argument
                annotator = BoxAnnotator(color=palette, text_scale=0.7, text_thickness=1, text_padding=2)
                annotated_img = annotator.annotate(
                    scene=img_np,
                    detections=detections,
                    labels=[f"{name}" for name in class_names]
                )

                # Resize the annotated image to a uniform, smaller size for display
                from PIL import Image as PILImage
                max_display_width = 500  # px, adjust as needed for your UI
                max_display_height = 500  # px, adjust as needed for your UI

                # Convert numpy array back to PIL Image for resizing
                annotated_pil = PILImage.fromarray(annotated_img)
                # Use "Resampling.LANCZOS" for Pillow >= 10, fallback to "LANCZOS" for older
                try:
                    resample = getattr(PILImage, "Resampling", PILImage).LANCZOS
                except AttributeError:
                    resample = PILImage.LANCZOS
                annotated_pil.thumbnail((max_display_width, max_display_height), resample)
                st.image(annotated_pil, use_container_width=False)
            else:
                st.warning("No annotation found for this image.")
        except Exception as e:
            st.write(f"Error displaying annotated image: {e}")
    else:
        st.info("Select an image to view with annotations.")