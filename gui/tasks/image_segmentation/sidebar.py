import os

import streamlit as st

from gui.tasks.utils import browse_file, browse_folder


IMAGE_SEGMENTATION_DATASETS = [
    "Cityscapes",
    "NuImages",
    "GAIA",
    "Generic",
    "Wildscenes",
    "RUGD",
    "Rellis3D",
    "GOOSE",
]

MODEL_INPUT_DATASETS = ["Cityscapes", "NuImages"]


def browse_dataset_path():
    """Callback for updating the dataset path text input."""
    folder = browse_folder()
    if folder:
        st.session_state.dataset_path = folder


def browse_segmentation_model_path():
    """Callback for updating the segmentation model path text input."""
    if st.session_state.get("segmentation_model_type") == "Hugging Face SegFormer":
        path = browse_folder()
    else:
        path = browse_file()

    if path:
        st.session_state.segmentation_model_path = path


def browse_segmentation_config_path():
    """Callback for updating the segmentation config path text input."""
    path = browse_file()
    if path:
        st.session_state.segmentation_config_path = path


def browse_segmentation_ontology_path():
    """Callback for updating the segmentation ontology path text input."""
    path = browse_file()
    if path:
        st.session_state.segmentation_ontology_path = path


def render_image_segmentation_sidebar(_available_devices):
    """Render the sidebar for the image segmentation task in Streamlit.
    :param _available_devices: List of available devices for model inference.
    :type _available_devices: list
    """
    with st.expander("Image Segmentation Dataset", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            dataset_type = st.selectbox(
                "Type",
                IMAGE_SEGMENTATION_DATASETS,
                key="segmentation_dataset_type",
            )
        with col2:
            st.selectbox("Split", ["train", "val", "test"], key="split")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_input("Dataset Folder", key="dataset_path")
        with col2:
            st.markdown(
                "<div style='margin-bottom: 1.75rem;'></div>",
                unsafe_allow_html=True,
            )
            st.button(
                "Browse",
                on_click=browse_dataset_path,
                key="browse_segmentation_dataset_path",
            )

        if dataset_type == "Cityscapes":
            render_cityscapes_dataset_inputs()
        elif dataset_type == "NuImages":
            render_nuimages_dataset_inputs()
        else:
            st.info(f"{dataset_type} image segmentation inputs are not wired yet.")

    with st.expander("Image Segmentation Model", expanded=False):
        dataset_type = st.session_state.get("segmentation_dataset_type", "Cityscapes")
        if dataset_type not in MODEL_INPUT_DATASETS:
            st.info(
                f"Image segmentation model loading is not wired for {dataset_type} yet."
            )
            return

        render_segmentation_model_inputs()

        if st.button(
            "Load Segmentation Model",
            type="primary",
            width="stretch",
            key="sidebar_load_segmentation_model_btn",
        ):
            load_image_segmentation_model()


def render_cityscapes_dataset_inputs():
    """Render the input fields for the Cityscapes dataset in the sidebar."""
    st.text_input(
        "Image Directory",
        value="leftImg8bit_trainvaltest/leftImg8bit",
        key="segmentation_image_dir",
    )
    st.text_input(
        "Label Directory",
        value="gtFine",
        key="segmentation_label_dir",
    )
    st.text_input(
        "Image Suffix",
        value="_leftImg8bit.png",
        key="segmentation_image_suffix",
    )
    st.text_input(
        "Label Suffix",
        value="_gtFine_labelIds.png",
        key="segmentation_label_suffix",
    )
    st.checkbox(
        "Use Train IDs",
        value=False,
        key="segmentation_use_train_id",
        help="Enable when labels use _gtFine_labelTrainIds.png.",
    )


def render_nuimages_dataset_inputs():
    """Render the input fields for the NuImages dataset in the sidebar."""
    st.text_input(
        "Version",
        value="v1.0-mini",
        key="nuimages_segmentation_version",
        help="nuImages version, for example v1.0-mini, v1.0-train, or v1.0-val.",
    )
    st.text_input(
        "Generated Labels Directory",
        value="generated/nuimages_segmentation_labels",
        key="nuimages_segmentation_labels_dir",
        help="Relative directory where generated segmentation masks are stored.",
    )


def render_segmentation_model_inputs():
    """Render the input fields for the image segmentation model in the sidebar."""
    model_type = st.selectbox(
        "Model Type",
        ["Torch Model File", "Hugging Face SegFormer"],
        key="segmentation_model_type",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_input(
            (
                "Model Path"
                if model_type == "Torch Model File"
                else "Model Name or Folder"
            ),
            key="segmentation_model_path",
            help=(
                "Path to a TorchScript model or saved PyTorch model file."
                if model_type == "Torch Model File"
                else "Hugging Face model name or local folder downloaded with save_pretrained."
            ),
        )
    with col2:
        st.markdown(
            "<div style='margin-bottom: 1.75rem;'></div>",
            unsafe_allow_html=True,
        )
        st.button(
            "Browse",
            on_click=browse_segmentation_model_path,
            key="browse_segmentation_model_path",
        )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_input(
            "Config File",
            key="segmentation_config_path",
            help="JSON model configuration file.",
        )
    with col2:
        st.markdown(
            "<div style='margin-bottom: 1.75rem;'></div>",
            unsafe_allow_html=True,
        )
        st.button(
            "Browse",
            on_click=browse_segmentation_config_path,
            key="browse_segmentation_config_path",
        )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_input(
            "Ontology File",
            key="segmentation_ontology_path",
            help="JSON file containing the model output ontology.",
        )
    with col2:
        st.markdown(
            "<div style='margin-bottom: 1.75rem;'></div>",
            unsafe_allow_html=True,
        )
        st.button(
            "Browse",
            on_click=browse_segmentation_ontology_path,
            key="browse_segmentation_ontology_path",
        )


def load_image_segmentation_model():
    """Render the image segmentation model in the sidebar."""
    from perceptionmetrics.models.torch_segmentation import TorchImageSegmentationModel

    model_type = st.session_state.get("segmentation_model_type", "Torch Model File")
    model_path = st.session_state.get("segmentation_model_path", "")
    config_path = st.session_state.get("segmentation_config_path", "")
    ontology_path = st.session_state.get("segmentation_ontology_path", "")

    if not model_path:
        st.error("Please provide a model path or model name.")
        return
    if not config_path or not os.path.isfile(config_path):
        st.error("Please provide a valid config JSON path.")
        return
    if not ontology_path or not os.path.isfile(ontology_path):
        st.error("Please provide a valid ontology JSON path.")
        return

    with st.spinner("Loading image segmentation model..."):
        try:
            model = load_model_for_type(model_type, model_path)
            segmentation_model = TorchImageSegmentationModel(
                model=model,
                model_cfg=config_path,
                ontology_fname=ontology_path,
            )
            st.session_state.segmentation_model = segmentation_model
            st.session_state.segmentation_model_loaded = True
            st.success("Segmentation model loaded and saved for inference")
        except Exception as exc:
            st.session_state.segmentation_model = None
            st.session_state.segmentation_model_loaded = False
            st.error(f"Failed to load segmentation model: {exc}")


def load_model_for_type(model_type, model_path):
    """Load a model based on the specified type and path.
    :param model_type: Type of the model to load (e.g., "Torch Model File", "Hugging Face SegFormer").
    :type model_type: str
    :param model_path: Path to the model file or directory.
    :type model_path: str
    :return: Loaded model instance.
    :rtype: object
    """
    if model_type == "Torch Model File":
        if not os.path.isfile(model_path):
            raise ValueError(
                "Torch Model File expects a .pt/.pth/.torchscript file saved with "
                "torch.save or torch.jit.save. For a Hugging Face model folder, "
                "select 'Hugging Face SegFormer'."
            )
        return model_path

    if model_type == "Hugging Face SegFormer":
        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError as exc:
            raise ImportError(
                "transformers is required for Hugging Face SegFormer models."
            ) from exc
        return SegformerForSemanticSegmentation.from_pretrained(model_path)

    raise ValueError(f"Unsupported segmentation model type: {model_type}")
