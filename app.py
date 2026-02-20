import streamlit as st
from tabs.dataset_viewer import dataset_viewer_tab
from tabs.inference import inference_tab
from tabs.evaluator import evaluator_tab
from perceptionmetrics.utils.gui import browse_folder


def browse_dataset_path():
    st.session_state.dataset_path = browse_folder()


st.set_page_config(page_title="PerceptionMetrics", layout="wide")

PAGES = {
    "Dataset Viewer": dataset_viewer_tab,
    "Inference": inference_tab,
    "Evaluator": evaluator_tab,
}

# Initialize commonly used session state keys
st.session_state.setdefault("dataset_path", "")
st.session_state.setdefault("dataset_type", "YOLO")
st.session_state.setdefault("split", "test")
st.session_state.setdefault("config_option", "Manual Configuration")
st.session_state.setdefault("confidence_threshold", 0.5)
st.session_state.setdefault("nms_threshold", 0.5)
st.session_state.setdefault("max_detections", 100)
st.session_state.setdefault("device", "cuda")
st.session_state.setdefault("batch_size", 1)
st.session_state.setdefault("evaluation_step", 5)
st.session_state.setdefault("detection_model", None)
st.session_state.setdefault("detection_model_loaded", False)

# Sidebar: Dataset Inputs
with st.sidebar:
    with st.expander("Dataset Inputs", expanded=True):
        # First row: Type and Split
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox(
                "Type",
                ["COCO", "YOLO"],
                key="dataset_type",
            )
        with col2:
            st.selectbox(
                "Split",
                ["train", "val", "test"],
                key="split",
            )

        # Second row: Path and Browse button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_input("Dataset Folder", key="dataset_path")
        with col2:
            st.markdown(
                "<div style='margin-bottom: 1.75rem;'></div>", unsafe_allow_html=True
            )
            st.button("Browse", on_click=browse_dataset_path)

        # Additional input for YOLO config file
        if st.session_state.get("dataset_type", "COCO") == "YOLO":
            st.file_uploader(
                "Dataset Configuration (.yaml)",
                type=["yaml"],
                key="dataset_config_file",
                help="Upload a YAML dataset configuration file.",
            )

    with st.expander("Model Inputs", expanded=False):
        st.file_uploader(
            "Model File (.pt, .onnx, .h5, .pb, .pth, .torchscript)",
            type=["pt", "onnx", "h5", "pb", "pth", "torchscript"],
            key="model_file",
            help="Upload your trained model file.",
            max_upload_size=1024,  # MB
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
        )
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
                    min_value=1,
                    max_value=1000,
                    step=1,
                    key="max_detections",
                )
            with col2:
                st.selectbox(
                    "Device",
                    ["cpu", "cuda", "mps"],
                    key="device",
                )
                st.selectbox(
                    "Model Format",
                    ["torchvision", "YOLO"],
                    index=(
                        0
                        if st.session_state.get("model_format", "torchvision")
                        == "torchvision"
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

            # Resize Logic
            enable_resize = st.checkbox(
                "Enable Resize", value=True, key="enable_resize"
            )

            if enable_resize:
                resize_strategy = st.radio(
                    "Resize Strategy",
                    ["Fixed Dimensions", "Min Side"],
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
                else:
                    st.number_input(
                        "Min Side",
                        min_value=1,
                        max_value=4096,
                        value=640,
                        step=1,
                        key="min_side",
                        help="Minimum size of the shorter side of the image",
                    )

            # Crop Logic
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

        # Load model action in sidebar
        from perceptionmetrics.models.torch_detection import TorchImageDetectionModel
        import json, tempfile

        load_model_btn = st.button(
            "Load Model",
            type="primary",
            width="stretch",
            help="Load and save the model for use in the Inference tab",
            key="sidebar_load_model_btn",
        )

        if load_model_btn:
            model_file = st.session_state.get("model_file")
            ontology_file = st.session_state.get("ontology_file")
            config_option = st.session_state.get(
                "config_option", "Manual Configuration"
            )
            config_file = (
                st.session_state.get("config_file")
                if config_option == "Upload Config File"
                else None
            )

            # Prepare configuration
            config_data = None
            config_path = None
            try:
                if config_option == "Upload Config File":
                    if config_file is not None:
                        config_data = json.load(config_file)
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".json", mode="w"
                        ) as tmp_cfg:
                            json.dump(config_data, tmp_cfg)
                            config_path = tmp_cfg.name
                    else:
                        st.error("Please upload a configuration file")
                else:
                    confidence_threshold = float(
                        st.session_state.get("confidence_threshold", 0.5)
                    )
                    nms_threshold = float(st.session_state.get("nms_threshold", 0.5))
                    max_detections = int(st.session_state.get("max_detections", 100))
                    device = st.session_state.get("device", "cpu")
                    batch_size = int(st.session_state.get("batch_size", 1))
                    evaluation_step = int(st.session_state.get("evaluation_step", 5))
                    model_format = st.session_state.get("model_format", "torchvision")

                    # Resize Logic extraction
                    enable_resize = st.session_state.get("enable_resize", True)
                    resize_cfg = None
                    if enable_resize:
                        resize_strategy = st.session_state.get(
                            "resize_strategy", "Fixed Dimensions"
                        )
                        if resize_strategy == "Fixed Dimensions":
                            resize_height = int(
                                st.session_state.get("resize_height", 640)
                            )
                            resize_width = int(
                                st.session_state.get("resize_width", 640)
                            )
                            resize_cfg = {
                                "height": resize_height,
                                "width": resize_width,
                            }
                        else:
                            min_side = int(st.session_state.get("min_side", 640))
                            resize_cfg = {"min_side": min_side}

                    config_data = {
                        "confidence_threshold": confidence_threshold,
                        "nms_threshold": nms_threshold,
                        "max_detections_per_image": max_detections,
                        "device": device,
                        "batch_size": batch_size,
                        "evaluation_step": evaluation_step,
                        "model_format": model_format.lower(),
                    }
                    if resize_cfg is not None:
                        config_data["resize"] = resize_cfg

                    if enable_crop:
                        crop_height = int(st.session_state.get("crop_height", 640))
                        crop_width = int(st.session_state.get("crop_width", 640))
                        crop_cfg = {"height": crop_height, "width": crop_width}
                        config_data["crop"] = crop_cfg

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".json", mode="w"
                    ) as tmp_cfg:
                        json.dump(config_data, tmp_cfg)
                        config_path = tmp_cfg.name
            except Exception as e:
                st.error(f"Failed to prepare configuration: {e}")
                config_path = None

            if model_file is None:
                st.error("Please upload a model file")
            elif config_path is None:
                st.error("Please provide a valid model configuration")
            elif ontology_file is None:
                st.error("Please upload an ontology file")
            else:
                with st.spinner("Loading model..."):
                    # Persist ontology to temp file
                    try:
                        ontology_data = json.load(ontology_file)
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".json", mode="w"
                        ) as tmp_ont:
                            json.dump(ontology_data, tmp_ont)
                            ontology_path = tmp_ont.name
                    except Exception as e:
                        st.error(f"Failed to load ontology: {e}")
                        ontology_path = None

                    # Persist model to temp file
                    try:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pt", mode="wb"
                        ) as tmp_model:
                            tmp_model.write(model_file.read())
                            model_temp_path = tmp_model.name
                    except Exception as e:
                        st.error(f"Failed to save model file: {e}")
                        model_temp_path = None

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

# Main content area with horizontal tabs
tab1, tab2, tab3 = st.tabs(["Dataset Viewer", "Inference", "Evaluator"])

with tab1:
    dataset_viewer_tab()
with tab2:
    inference_tab()
with tab3:
    evaluator_tab()
