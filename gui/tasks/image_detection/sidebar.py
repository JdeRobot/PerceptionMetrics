import json
import os
import tempfile
from typing import Optional

import streamlit as st

from gui.tasks.utils import browse_folder


def browse_dataset_path():
    folder = browse_folder()
    if folder:
        st.session_state.dataset_path = folder


def render_image_detection_sidebar(available_devices):
    """Render the sidebar for the image detection task in Streamlit.
    :param available_devices: List of available devices for model inference
    :type available_devices: list
    """

    with st.expander("Image Detection Dataset", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox(
                "Type", ["COCO", "YOLO"], key="dataset_type"
            )  # split into two columns
        with col2:
            st.selectbox("Split", ["train", "val", "test"], key="split")

        col1, col2 = st.columns([3.5, 2.5])
        with col1:
            st.text_input("Dataset Folder", key="dataset_path")
        with col2:
            st.markdown(
                "<div style='margin-bottom: 1.75rem;'></div>",  # add some spacing to align with the text input
                unsafe_allow_html=True,
            )
            st.button(
                "Browse",
                on_click=browse_dataset_path,
                key="browse_detection_dataset_path",
                use_container_width=True,
            )

        if st.session_state.get("dataset_type", "COCO") == "YOLO":
            st.file_uploader(
                "Dataset Configuration (.yaml)",
                type=["yaml"],
                key="dataset_config_file",
                help="Upload a YAML dataset configuration file.",
            )

    with st.expander("Image Detection Model", expanded=False):
        st.file_uploader(
            "Model File (.pt, .onnx, .h5, .pb, .pth, .torchscript)",
            type=["pt", "onnx", "h5", "pb", "pth", "torchscript"],
            key="model_file",
            help="Upload your trained model file.",
            max_upload_size=1024,
        )
        st.file_uploader(
            "Ontology File (.json)",
            type=["json"],
            key="ontology_file",
            help="Upload a JSON file with class labels.",
        )
        st.radio(
            "Configuration Method:",
            ["Manual Configuration", "Upload Config File"],
            key="config_option",
            horizontal=True,
        )  # radio button to select between manual configuration and uploading a config file
        if (
            st.session_state.get("config_option", "Manual Configuration")
            == "Upload Config File"
        ):
            st.file_uploader(
                "Configuration File (.json)",
                type=["json"],
                key="config_file",
                help="Upload a JSON configuration file.",
            )
        else:
            _render_manual_detection_model_config(available_devices)

        if st.button(
            "Load Model",
            type="primary",
            width="stretch",
            help="Load and save the model for use in the Inference tab",
            key="sidebar_load_model_btn",
        ):
            load_image_detection_model()


def _render_manual_detection_model_config(available_devices):
    """Render the manual configuration options for the image detection model in the sidebar.
    :param available_devices: List of available devices for model inference
    :type available_devices: list
    """
    col1, col2 = st.columns(2)
    with col1:
        st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="confidence_threshold",
            help="Minimum confidence score for detections",
        )
        st.slider(
            "NMS Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="nms_threshold",
            help="Non-maximum suppression threshold",
        )
        st.number_input(
            "Max Detections/Image",
            min_value=-1,
            max_value=1000,
            step=1,
            key="max_detections",
        )
    with col2:
        st.selectbox("Device", available_devices, key="device")
        st.selectbox(
            "Model Format",
            ["torchvision", "YOLO"],
            index=(
                0
                if st.session_state.get("model_format", "torchvision") == "torchvision"
                else 1
            ),
            key="model_format",
        )
        st.number_input(
            "Batch Size",
            min_value=1,
            max_value=256,
            step=1,
            key="batch_size",
        )
        st.number_input(
            "Evaluation Step",
            min_value=0,
            max_value=1000,
            step=1,
            key="evaluation_step",
            help="Update UI with intermediate metrics every N images (0 = disable intermediate updates)",
        )

    st.write("---")
    st.write("**Image Size Configuration**")

    enable_resize = st.checkbox("Enable Resize", value=True, key="enable_resize")

    if enable_resize:
        resize_strategy = st.radio(
            "Resize Strategy",
            ["Min Side", "Max Side", "Fixed Dimensions"],
            key="resize_strategy",
            horizontal=True,
            label_visibility="collapsed",
        )

        if resize_strategy == "Fixed Dimensions":
            c1, c2 = st.columns(2)
            with c1:
                st.number_input(
                    "Image Resize Height",
                    min_value=1,
                    max_value=4096,
                    value=640,
                    step=1,
                    key="resize_height",
                    help="Height to resize images for inference",
                )
            with c2:
                st.number_input(
                    "Image Resize Width",
                    min_value=1,
                    max_value=4096,
                    value=640,
                    step=1,
                    key="resize_width",
                    help="Width to resize images for inference",
                )
        elif resize_strategy == "Min Side":
            st.number_input(
                "Min Side",
                min_value=1,
                max_value=4096,
                value=640,
                step=1,
                key="min_side",
                help="Minimum size of the shorter side of the image",
            )
        elif resize_strategy == "Max Side":
            st.number_input(
                "Max Side",
                min_value=1,
                max_value=4096,
                value=640,
                step=1,
                key="max_side",
                help="Maximum size of the longer side of the image",
            )
        else:
            st.error("Invalid resize strategy selected")

    enable_pad = st.checkbox(
        "Enable Padding to Closest Multiple", value=True, key="enable_pad"
    )

    if enable_pad:
        st.number_input(
            "Divisor",
            min_value=1,
            max_value=128,
            value=32,
            step=1,
            key="pad_divisor",
            help="Pad image dimensions to the closest multiple of this value",
        )

    enable_crop = st.checkbox("Enable Center Crop", key="enable_crop")

    if enable_crop:
        c1, c2 = st.columns(2)
        with c1:
            st.number_input(
                "Crop Height",
                min_value=1,
                max_value=4096,
                value=640,
                step=1,
                key="crop_height",
                help="Center crop height",
            )
        with c2:
            st.number_input(
                "Crop Width",
                min_value=1,
                max_value=4096,
                value=640,
                step=1,
                key="crop_width",
                help="Center crop width",
            )


def load_image_detection_model():
    """Load the image detection model based on the provided configuration and ontology files."""
    from perceptionmetrics.models.torch_detection import TorchImageDetectionModel

    model_file = st.session_state.get("model_file")
    ontology_file = st.session_state.get("ontology_file")
    config_path = _write_detection_config()

    if model_file is None:
        st.error("Please upload a model file")
    elif config_path is None:
        st.error("Please provide a valid model configuration")
    elif ontology_file is None:
        st.error("Please upload an ontology file")
    else:
        with st.spinner("Loading model..."):
            ontology_path = _uploaded_json_to_tempfile(ontology_file)
            model_temp_path = _uploaded_model_to_tempfile(model_file)

            if ontology_path and model_temp_path:
                try:
                    model = TorchImageDetectionModel(
                        model=model_temp_path,
                        model_cfg=config_path,
                        ontology_fname=ontology_path,
                        device=st.session_state.get("device", "cpu"),
                    )
                    st.session_state.detection_model = model
                    st.session_state.detection_model_loaded = True
                    st.success("Model loaded and saved for inference")
                except Exception as e:
                    st.session_state.detection_model = None
                    st.session_state.detection_model_loaded = False
                    st.error(f"Failed to load model: {e}")


def _write_detection_config() -> Optional[str]:
    """Write the detection configuration to a temporary JSON file based on the selected configuration method.
    :return: Path to the temporary JSON configuration file or None if an error occurred
    :rtype: Optional[str]
    """
    config_option = st.session_state.get("config_option", "Manual Configuration")
    config_file = (
        st.session_state.get("config_file")
        if config_option == "Upload Config File"
        else None
    )

    try:
        if config_option == "Upload Config File":
            if config_file is None:
                st.error("Please upload a configuration file")
                return None
            config_data = json.load(config_file)
        else:
            config_data = _manual_detection_config()

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".json", mode="w"
        ) as tmp_cfg:
            json.dump(config_data, tmp_cfg)
            return tmp_cfg.name
    except Exception as e:
        st.error(f"Failed to prepare configuration: {e}")
        return None


def _manual_detection_config() -> dict:
    """Generate a configuration dictionary based on the manual configuration options in the sidebar.
    :return: Configuration dictionary
    :rtype: dict
    """
    resize_cfg = None
    if st.session_state.get("enable_resize", True):
        resize_strategy = st.session_state.get("resize_strategy", "Fixed Dimensions")
        if resize_strategy == "Fixed Dimensions":
            resize_cfg = {
                "height": int(st.session_state.get("resize_height", 640)),
                "width": int(st.session_state.get("resize_width", 640)),
            }
        elif resize_strategy == "Min Side":
            resize_cfg = {"min_side": int(st.session_state.get("min_side", 640))}
        elif resize_strategy == "Max Side":
            resize_cfg = {"max_side": int(st.session_state.get("max_side", 640))}

    config_data = {
        "confidence_threshold": float(
            st.session_state.get("confidence_threshold", 0.5)
        ),
        "nms_threshold": float(st.session_state.get("nms_threshold", 0.5)),
        "max_detections_per_image": int(st.session_state.get("max_detections", -1)),
        "device": st.session_state.get("device", "cpu"),
        "batch_size": int(st.session_state.get("batch_size", 1)),
        "evaluation_step": int(st.session_state.get("evaluation_step", 5)),
        "model_format": st.session_state.get("model_format", "torchvision").lower(),
    }
    if resize_cfg is not None:
        config_data["resize"] = resize_cfg

    if st.session_state.get("enable_pad", True):
        pad_divisor = int(st.session_state.get("pad_divisor", 32))
        if resize_cfg is not None:
            resize_cfg["closest_divisor"] = pad_divisor
        else:
            config_data["resize"] = {"closest_divisor": pad_divisor}

    if st.session_state.get("enable_crop", False):
        config_data["crop"] = {
            "height": int(st.session_state.get("crop_height", 640)),
            "width": int(st.session_state.get("crop_width", 640)),
        }

    return config_data


def _uploaded_json_to_tempfile(uploaded_file) -> Optional[str]:
    """Save the uploaded JSON file to a temporary file and return its path.
    :param uploaded_file: Uploaded JSON file
    :type uploaded_file: UploadedFile
    :return: Path to the temporary JSON file or None if an error occurred
    :rtype: Optional[str]
    """

    try:
        data = json.load(uploaded_file)
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".json", mode="w"
        ) as tmp_file:
            json.dump(data, tmp_file)
            return tmp_file.name
    except Exception as e:
        st.error(f"Failed to load JSON file: {e}")
        return None


def _uploaded_model_to_tempfile(uploaded_file) -> Optional[str]:
    """Save the uploaded model file to a temporary file and return its path.
    :param uploaded_file: Uploaded model file
    :type uploaded_file: UploadedFile
    :return: Path to the temporary model file or None if an error occurred
    :rtype: Optional[str]
    """
    try:
        suffix = os.path.splitext(uploaded_file.name)[1] or ".pt"
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, mode="wb"
        ) as tmp_model:
            tmp_model.write(uploaded_file.read())
            return tmp_model.name
    except Exception as e:
        st.error(f"Failed to save model file: {e}")
        return None
