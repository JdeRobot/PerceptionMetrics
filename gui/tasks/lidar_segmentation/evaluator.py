import json
import os
import tempfile

import streamlit as st

from gui.tasks.lidar_segmentation.dataset_viewer import load_semantic_kitti_dataset
from gui.tasks.utils import browse_folder


def browse_lidar_predictions_outdir():
    folder = browse_folder()
    if folder:
        st.session_state.lidar_predictions_outdir = folder


def render_lidar_segmentation_evaluator():
    st.header("Evaluator")
    st.markdown("Evaluate your LiDAR segmentation model on SemanticKITTI.")

    dataset_type = st.session_state.get("lidar_dataset_type", "SemanticKITTI")
    if dataset_type != "SemanticKITTI":
        st.info(f"{dataset_type} LiDAR segmentation evaluation is not wired yet.")
        return

    model = st.session_state.get("lidar_model")
    dataset = None
    dataset_path = st.session_state.get("dataset_path", "")
    config_path = st.session_state.get("lidar_config_path", "")
    split = st.session_state.get("split", "val")

    if not dataset_path or not os.path.isdir(dataset_path):
        st.warning("No dataset path provided. Please set the dataset path in the sidebar.")
    elif not config_path or not os.path.isfile(config_path):
        st.warning("No SemanticKITTI config YAML provided. Please set it in the sidebar.")
    else:
        try:
            dataset = load_semantic_kitti_dataset(dataset_path, config_path, split)
            st.success(
                f"✅ Dataset loaded: {dataset_path} ({split} split) - "
                f"{len(dataset.dataset)} samples"
            )
        except Exception as exc:
            st.error(f"Error loading SemanticKITTI dataset: {exc}")

    if model is not None:
        st.success("✅ LiDAR model loaded and ready for evaluation")
    else:
        st.warning(
            "No LiDAR model loaded. Please load a model using the "
            "'Load LiDAR Model' button in the sidebar."
        )

    st.markdown("### Evaluation Configuration")

    save_predictions = st.checkbox(
        "Save Predictions",
        value=False,
        help="Save predicted label files to an output directory.",
        key="lidar_save_predictions",
    )

    predictions_outdir = None
    if save_predictions:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_input(
                "Predictions Output Directory",
                key="lidar_predictions_outdir",
            )
        with col2:
            st.markdown(
                "<div style='margin-bottom: 1.75rem;'></div>",
                unsafe_allow_html=True,
            )
            st.button(
                "Browse",
                on_click=browse_lidar_predictions_outdir,
                key="browse_lidar_predictions_outdir",
            )
        predictions_outdir = st.session_state.get("lidar_predictions_outdir")

    ontology_translation = st.file_uploader(
        "Ontology Translation (Optional)",
        type=["json"],
        key="lidar_ontology_translation",
        help="JSON file for translating between dataset and model ontologies.",
    )

    st.info(
        "For the MMDetection3D SemanticKITTI tutorial model, upload "
        "semantickitti_raw_to_mmdet3d_translation.json and use dataset_to_model."
    )

    translation_direction = st.selectbox(
        "Translation Direction",
        ["dataset_to_model", "model_to_dataset"],
        key="lidar_translation_direction",
        help=(
            "dataset_to_model maps GT labels to model IDs. "
            "model_to_dataset maps predictions to dataset IDs."
        ),
    )

    output_dir_missing = save_predictions and not (
        predictions_outdir and predictions_outdir.strip()
    )
    if output_dir_missing:
        st.warning("Please provide a predictions output directory.")

    if st.button(
        "🚀 Run Evaluation",
        type="primary",
        disabled=dataset is None or model is None or output_dir_missing,
        key="run_lidar_evaluation",
    ):
        with st.spinner("Running LiDAR evaluation..."):
            try:
                ontology_translation_path = None
                if ontology_translation is not None:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".json", mode="w"
                    ) as tmp_trans:
                        json.dump(json.load(ontology_translation), tmp_trans)
                        ontology_translation_path = tmp_trans.name

                predictions_outdir = (
                    predictions_outdir.strip()
                    if (save_predictions and predictions_outdir)
                    else None
                )
                if predictions_outdir is not None:
                    os.makedirs(predictions_outdir, exist_ok=True)

                progress_bar = st.progress(0)
                status_text = st.empty()
                intermediate_metrics_placeholder = st.empty()

                def progress_callback(processed, total):
                    progress = processed / total if total > 0 else 0
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Processing: {processed}/{total} point clouds ({progress:.1%})"
                    )

                def metrics_callback(metrics_df, processed, total):
                    with intermediate_metrics_placeholder.container():
                        st.markdown(
                            f"#### Results (after {processed}/{total} point clouds)"
                        )
                        display_lidar_evaluation_results(
                            metrics_df, show_download=False
                        )

                results = model.eval(
                    dataset=dataset,
                    split=split,
                    ontology_translation=ontology_translation_path,
                    translation_direction=translation_direction,
                    predictions_outdir=predictions_outdir,
                    results_per_sample=save_predictions,
                    progress_callback=progress_callback,
                    metrics_callback=metrics_callback,
                )

                progress_bar.empty()
                status_text.empty()
                intermediate_metrics_placeholder.empty()

                st.session_state["lidar_evaluation_results"] = results
                st.success("✅ Evaluation completed successfully!")
            except Exception as exc:
                st.error(f"Error in model.eval(): {exc}")

    if "lidar_evaluation_results" in st.session_state:
        display_lidar_evaluation_results(
            st.session_state["lidar_evaluation_results"]
        )


def display_lidar_evaluation_results(results, show_download=True):
    if results is None or results.empty:
        st.warning("No evaluation results to display.")
        return

    st.markdown("#### Metrics")
    display_df = results.copy()
    numeric_columns = display_df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_columns:
        display_df[col] = display_df[col].round(3)
    st.dataframe(display_df, width="stretch")

    if show_download:
        csv = results.to_csv(index=True)
        st.download_button(
            label="📥 Download LiDAR segmentation metrics",
            data=csv,
            file_name="lidar_segmentation_evaluation_results.csv",
            mime="text/csv",
        )
