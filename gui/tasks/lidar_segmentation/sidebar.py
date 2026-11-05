import streamlit as st

from gui.tasks.utils import browse_file, browse_folder


def browse_lidar_dataset_path():
    """ Callback function to browse for the LiDAR dataset path. """
    folder = browse_folder()
    if folder:
        st.session_state.dataset_path = folder


def browse_lidar_config_path():
    """ Callback function to browse for the LiDAR config path. """
    file_path = browse_file()
    if file_path:
        st.session_state.lidar_config_path = file_path


def browse_lidar_mmdet3d_config_path():
    """ Callback function to browse for the LiDAR MMDetection3D config path. """
    file_path = browse_file()
    if file_path:
        st.session_state.lidar_mmdet3d_config_path = file_path


def browse_lidar_checkpoint_path():
    """ Callback function to browse for the LiDAR checkpoint path. """
    file_path = browse_file()
    if file_path:
        st.session_state.lidar_checkpoint_path = file_path


def browse_lidar_model_config_path():
    """ Callback function to browse for the LiDAR model config path. """
    file_path = browse_file()
    if file_path:
        st.session_state.lidar_model_config_path = file_path


def browse_lidar_ontology_path():
    """ Callback function to browse for the LiDAR ontology path. """
    file_path = browse_file()
    if file_path:
        st.session_state.lidar_ontology_path = file_path

def render_lidar_path_input(label, key, browse_callback, help_text):
    """ Render a text input with a browse button for LiDAR paths.
    :param label: Label for the text input.
    :type label: str
    :param key: Key for the text input in Streamlit session state.
    :type key: str
    :param browse_callback: Callback function to be called when the browse button is clicked.
    :type browse_callback: function
    :param help_text: Help text to be displayed for the text input.
    :type help_text: str"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_input(label, key=key, help=help_text)
    with col2:
        st.markdown(
            "<div style='margin-bottom: 1.75rem;'></div>",
            unsafe_allow_html=True,
        )
        st.button(
            "Browse",
            on_click=browse_callback,
            key=f"browse_{key}",
        )

def render_lidar_segmentation_sidebar(_available_devices):
    """ Render the sidebar for LiDAR segmentation.
    :param _available_devices: List of available devices for model inference.
    :type _available_devices: list
    """
    
    with st.expander("LiDAR Segmentation Dataset", expanded=True):
        st.selectbox(
            "Type",
            ["SemanticKITTI"],
            key="lidar_dataset_type",
        )
        st.selectbox("Split", ["train", "val", "test"], key="split")

        render_lidar_path_input("Dataset Folder", 
                                "dataset_path",
                                browse_lidar_dataset_path,
                                "SemanticKITTI root folder containing the sequences directory."
        )

        render_lidar_path_input("SemanticKITTI Config YAML", 
                                "lidar_config_path",
                                browse_lidar_config_path, 
                                "semantic-kitti.yaml used for splits and label colors."
        )

    with st.expander("LiDAR Segmentation Model", expanded=False):
        st.selectbox(
            "Model Type",
            ["MMDetection3D"],
            key="lidar_model_type",
        )

        render_lidar_path_input(
            "MMDetection3D Config",
            "lidar_mmdet3d_config_path",
            browse_lidar_mmdet3d_config_path,
            "MMDetection3D model config Python file.",
        )
        render_lidar_path_input(
            "Checkpoint",
            "lidar_checkpoint_path",
            browse_lidar_checkpoint_path,
            "Model checkpoint file.",
        )
        render_lidar_path_input(
            "Model Config JSON",
            "lidar_model_config_path",
            browse_lidar_model_config_path,
            "PerceptionMetrics LiDAR model configuration JSON.",
        )
        render_lidar_path_input(
            "Ontology File",
            "lidar_ontology_path",
            browse_lidar_ontology_path,
            "JSON file containing the model output ontology.",
        )

        st.selectbox(
            "Device",
            _available_devices,
            key="device",
        )

        if st.button(
            "Load LiDAR Model",
            type="primary",
            width="stretch",
            key="sidebar_load_lidar_model_btn",
        ):
            load_lidar_segmentation_model()




def load_lidar_segmentation_model():
    """ Load the LiDAR segmentation model based on the provided configuration and checkpoint paths. """
    mmdet3d_config_path = st.session_state.get("lidar_mmdet3d_config_path", "")
    checkpoint_path = st.session_state.get("lidar_checkpoint_path", "")
    model_config_path = st.session_state.get("lidar_model_config_path", "")
    ontology_path = st.session_state.get("lidar_ontology_path", "")

    if not mmdet3d_config_path:
        st.error("Please provide the MMDetection3D config path.")
        return
    if not checkpoint_path:
        st.error("Please provide the checkpoint path.")
        return
    if not model_config_path:
        st.error("Please provide the model config JSON path.")
        return
    if not ontology_path:
        st.error("Please provide the model ontology path.")
        return

    device = st.session_state.get("device", "cpu")
    mmdet3d_device = "cuda:0" if device == "cuda" else device

    with st.spinner("Loading LiDAR segmentation model..."):
        try:
            from mmdet3d.apis import init_model
            from perceptionmetrics.models import TorchLiDARSegmentationModel

            mmdet3d_model = init_model(
                config=mmdet3d_config_path,
                checkpoint=checkpoint_path,
                device=mmdet3d_device,
            )
            lidar_model = TorchLiDARSegmentationModel(
                model=mmdet3d_model,
                model_cfg=model_config_path,
                ontology_fname=ontology_path,
            )
            st.session_state.lidar_model = lidar_model
            st.session_state.lidar_model_loaded = True
            st.success(
                "LiDAR segmentation model loaded and saved for inference")
        except Exception as exc:
            st.session_state.lidar_model = None
            st.session_state.lidar_model_loaded = False
            st.error(f"Failed to load LiDAR segmentation model: {exc}")
