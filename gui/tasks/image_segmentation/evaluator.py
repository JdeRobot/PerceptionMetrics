import json
import os
import tempfile

import streamlit as st

from gui.tasks.image_segmentation.dataset_viewer import (
    load_image_segmentation_dataset,
)
from gui.tasks.utils import browse_folder


def browse_segmentation_predictions_outdir():
    """Callback for updating the predictions output directory text input."""
    folder = browse_folder()
    if folder:
        st.session_state.segmentation_predictions_outdir = folder


def render_image_segmentation_evaluator():
    """Render the image segmentation evaluator tab in the Streamlit app."""
    st.header("Evaluator")
    st.markdown("Evaluate your model on the loaded dataset using PerceptionMetrics.")

    dataset_type = st.session_state.get("segmentation_dataset_type", "Cityscapes")
    if dataset_type not in ["Cityscapes", "NuImages"]:
        st.info(f"{dataset_type} image segmentation evaluation is not wired yet.")
        return

    dataset = None
    model = st.session_state.get("segmentation_model")
    dataset_path = st.session_state.get("dataset_path", "")
    split = st.session_state.get("split", "val")

    if not dataset_path or not os.path.isdir(dataset_path):
        st.warning(
            "No dataset path provided. Please set the dataset path in the sidebar."
        )
    elif dataset_type == "NuImages" and split == "test":
        st.warning("NuImages segmentation evaluation supports train and val splits.")
    else:
        try:
            dataset = load_image_segmentation_dataset(dataset_type, dataset_path, split)
            st.success(
                f"✅ Dataset loaded: {dataset_path} ({split} split) - {len(dataset.dataset)} samples"
            )
        except Exception as exc:
            st.error(f"Error loading dataset: {exc}")

    if model is not None:
        st.success("✅ Model loaded and ready for evaluation")
    else:
        st.warning(
            "No model loaded. Please load a model using the "
            "'Load Segmentation Model' button in the sidebar."
        )

    st.markdown("### Evaluation Configuration")

    save_predictions = st.checkbox(
        "Save Predictions",
        value=False,
        help="Save predicted label images to an output directory.",
        key="segmentation_save_predictions",
    )

    predictions_outdir = None
    if save_predictions:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_input(
                "Predictions Output Directory",
                key="segmentation_predictions_outdir",
            )
        with col2:
            st.markdown(
                "<div style='margin-bottom: 1.75rem;'></div>",
                unsafe_allow_html=True,
            )
            st.button(
                "Browse",
                on_click=browse_segmentation_predictions_outdir,
                key="browse_segmentation_predictions_outdir",
            )
        predictions_outdir = st.session_state.get("segmentation_predictions_outdir")

    ontology_translation = st.file_uploader(
        "Ontology Translation (Optional)",
        type=["json"],
        key="segmentation_ontology_translation",
        help="JSON file for translating between dataset and model ontologies.",
    )
    if dataset_type == "Cityscapes":
        st.info(
            "For Cityscapes SegFormer models, use train-ID labels or provide a "
            "label-ID to train-ID ontology translation."
        )
    elif dataset_type == "NuImages":
        st.info(
            "For NuImages SegFormer models, upload "
            "nuimages_to_model_ontology_translation.json and use dataset_to_model."
        )

    translation_direction = st.selectbox(
        "Translation Direction",
        ["dataset_to_model", "model_to_dataset"],
        key="segmentation_translation_direction",
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
        key="run_segmentation_evaluation",
    ):
        with st.spinner("Running evaluation..."):
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
                        f"Processing: {processed}/{total} images ({progress:.1%})"
                    )

                def metrics_callback(metrics_df, processed, total):
                    with intermediate_metrics_placeholder.container():
                        st.markdown(f"#### Results (after {processed}/{total} images)")
                        display_segmentation_evaluation_results(
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

                st.session_state["segmentation_evaluation_results"] = results
                st.success("✅ Evaluation completed successfully!")
            except Exception as exc:
                st.error(f"Error in model.eval(): {exc}")

    if "segmentation_evaluation_results" in st.session_state:
        display_segmentation_evaluation_results(
            st.session_state["segmentation_evaluation_results"]
        )


def display_segmentation_evaluation_results(results, show_download=True):
    """Display the evaluation results in a Streamlit dataframe and provide a download button.
    :param results: pd.DataFrame, the evaluation results to display
    :type results: pd.DataFrame
    :param show_download: bool, whether to show the download button for the results
    :type show_download: bool"""
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
            label="📥 Download segmentation metrics",
            data=csv,
            file_name="segmentation_evaluation_results.csv",
            mime="text/csv",
        )
