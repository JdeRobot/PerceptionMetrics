import streamlit as st
import os
import tempfile
import json
from perceptionmetrics.datasets.coco import CocoDataset


def evaluator_tab():
    st.header("Evaluator")
    st.markdown("Evaluate your model on the loaded dataset using PerceptionMetrics.")

    # Check if we have the required objects from sidebar inputs
    dataset_available = False
    model_available = False
    dataset = None
    model = None

    # Check for dataset from sidebar inputs
    dataset_path = st.session_state.get("dataset_path", "")
    dataset_type = st.session_state.get("dataset_type", "Coco")
    split = st.session_state.get("split", "val")

    # Try to get existing dataset from session state first
    dataset_key = f"{dataset_path}_{split}"
    if dataset_key in st.session_state:
        dataset = st.session_state[dataset_key]
        dataset_available = True
        st.success(
            f"✅ Dataset loaded: {dataset_path} ({split} split) - {len(dataset.dataset)} samples"
        )
    elif dataset_path and os.path.isdir(dataset_path):
        try:
            if dataset_type.lower() == "coco":
                img_dir = os.path.join(dataset_path, f"images/{split}2017")
                ann_file = os.path.join(
                    dataset_path, "annotations", f"instances_{split}2017.json"
                )

                if os.path.isdir(img_dir) and os.path.isfile(ann_file):
                    st.session_state[dataset_key] = CocoDataset(
                        annotation_file=ann_file, image_dir=img_dir, split=split
                    )
                    # Make filenames global - this is crucial for evaluation
                    st.session_state[dataset_key].make_fname_global()
                    dataset = st.session_state[dataset_key]
                    dataset_available = True
                    st.success(
                        f"✅ Dataset loaded: {dataset_path} ({split} split) - {len(dataset.dataset)} samples"
                    )
                else:
                    st.warning(
                        "⚠️ Dataset files not found. Please check the dataset path and split in the sidebar."
                    )
            else:
                st.warning(
                    "⚠️ Only COCO datasets are currently supported for evaluation."
                )
        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")
    else:
        st.warning(
            "⚠️ No dataset path provided. Please set the dataset path in the sidebar."
        )

    # Check for model from sidebar (loaded via Load Model button)
    if (
        "detection_model" in st.session_state
        and st.session_state.detection_model is not None
    ):
        model = st.session_state.detection_model
        model_available = True
        st.success("✅ Model loaded and ready for evaluation")
    else:
        st.warning(
            "⚠️ No model loaded. Please load a model using the 'Load Model' button in the sidebar."
        )

    # Evaluation configuration
    st.markdown("### Evaluation Configuration")

    save_predictions = st.checkbox(
        "Save Predictions",
        value=False,
        help="Save individual predictions and metrics per sample",
    )

    ontology_translation = st.file_uploader(
        "Ontology Translation (Optional)",
        type=["json"],
        help="JSON file for translating between dataset and model ontologies",
    )

    # Run evaluation button
    if st.button(
        "🚀 Run Evaluation",
        type="primary",
        disabled=not (dataset_available and model_available),
    ):
        if not dataset_available or not model_available:
            st.error(
                "Please ensure both dataset and model are loaded before running evaluation."
            )
            return

        # Prepare evaluation
        with st.spinner("Running evaluation..."):
            try:
                # Validate dataset and model
                if len(dataset.dataset) == 0:
                    st.error(
                        "Dataset has no samples. Please check the dataset configuration."
                    )
                    return

                if not hasattr(model, "model_cfg") or model.model_cfg is None:
                    st.error(
                        "Model configuration is missing. Please reload the model in the Inference tab."
                    )
                    return

                # Handle ontology translation if provided
                ontology_translation_path = None
                if ontology_translation is not None:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".json", mode="w"
                    ) as tmp_trans:
                        json.dump(json.load(ontology_translation), tmp_trans)
                        ontology_translation_path = tmp_trans.name

                # Prepare predictions output directory if needed
                predictions_outdir = None
                if save_predictions:
                    predictions_outdir = tempfile.mkdtemp(prefix="eval_predictions_")

                # Create progress bar for evaluation
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Create placeholders for intermediate metrics that will be updated in place
                intermediate_metrics_placeholder = st.empty()
                intermediate_table_placeholder = st.empty()

                def progress_callback(processed, total):
                    """Progress callback for Streamlit UI"""
                    try:
                        progress = processed / total if total > 0 else 0
                        progress_bar.progress(progress)
                        status_text.text(
                            f"Processing: {processed}/{total} images ({progress:.1%})"
                        )
                    except Exception as e:
                        st.error(f"Progress callback error: {e}")

                def metrics_callback(metrics_df, processed, total):
                    """Metrics callback for intermediate results display"""
                    try:
                        # Update the metrics placeholder with current summary metrics
                        if "mean" in metrics_df.columns:
                            mean_metrics = metrics_df["mean"]

                            with intermediate_metrics_placeholder.container():
                                st.markdown(
                                    f"#### 📊 Intermediate Results (after {processed} images)"
                                )

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("mAP", f"{mean_metrics.get('AP', 0):.3f}")
                                with col2:
                                    st.metric(
                                        "Mean Precision",
                                        f"{mean_metrics.get('Precision', 0):.3f}",
                                    )
                                with col3:
                                    st.metric(
                                        "Mean Recall",
                                        f"{mean_metrics.get('Recall', 0):.3f}",
                                    )

                        # Update the table placeholder with current per-class results
                        per_class_results = (
                            metrics_df.drop(columns=["mean"])
                            if "mean" in metrics_df.columns
                            else metrics_df
                        )
                        per_class_results = per_class_results.drop(
                            ["AUC-PR", "mAP@[0.5:0.95]"], errors="ignore"
                        )

                        # Round for display
                        display_df = per_class_results.copy()
                        numeric_columns = display_df.select_dtypes(
                            include=["float64", "int64"]
                        ).columns
                        for col in numeric_columns:
                            if col in display_df.columns:
                                display_df[col] = display_df[col].round(3)

                        with intermediate_table_placeholder.container():
                            st.markdown("#### Per-Class Metrics (Intermediate)")
                            st.dataframe(display_df, width="stretch")

                    except Exception as e:
                        st.error(f"Metrics callback error: {e}")

                # Run evaluation with progress tracking
                # Use full dataset for evaluation

                try:
                    # Use the full dataset for evaluation
                    results = model.eval(
                        dataset=dataset,
                        split=split,
                        ontology_translation=ontology_translation_path,
                        predictions_outdir=predictions_outdir,
                        results_per_sample=save_predictions,
                        progress_callback=progress_callback,
                        metrics_callback=metrics_callback,
                    )
                except Exception as e:
                    st.error(f"Error in model.eval(): {e}")
                    return

                # Results ready

                # Clear progress elements and intermediate results
                progress_bar.empty()
                status_text.empty()
                intermediate_metrics_placeholder.empty()
                intermediate_table_placeholder.empty()

                # Store results in session state
                st.session_state["evaluation_results"] = results
                st.session_state["evaluation_config"] = {
                    "split": split,
                    "predictions_saved": save_predictions,
                }

                st.success("✅ Evaluation completed successfully!")

            except Exception as e:
                st.error(f"❌ Evaluation failed: {e}")
                import traceback

                st.code(traceback.format_exc())

    # Display results (either from current evaluation or previous)
    if "evaluation_results" in st.session_state:
        display_evaluation_results(st.session_state["evaluation_results"])


def display_evaluation_results(results):
    """Display evaluation results in a comprehensive format"""

    if results is None:
        st.warning("No evaluation results to display.")
        return

    # Handle new results format (dictionary with metrics_df and metrics_factory)
    if isinstance(results, dict):
        metrics_df = results.get("metrics_df")
        metrics_factory = results.get("metrics_factory")
    else:
        # Fallback for old format
        metrics_df = results
        metrics_factory = None

    if metrics_df is None or metrics_df.empty:
        st.warning("No evaluation results to display.")
        return

    # Display summary metrics
    st.markdown("#### Summary Metrics")

    # Get mean metrics - mean is a column
    if "mean" in metrics_df.columns:
        mean_metrics = metrics_df["mean"]

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("mAP", f"{mean_metrics.get('AP', 0):.3f}")
        with col2:
            st.metric("Mean Precision", f"{mean_metrics.get('Precision', 0):.3f}")
        with col3:
            st.metric("Mean Recall", f"{mean_metrics.get('Recall', 0):.3f}")
        with col4:
            coco_map = mean_metrics.get("mAP@[0.5:0.95]", 0)
            st.metric("mAP@[0.5:0.95]", f"{coco_map:.3f}")
        with col5:
            auc_pr = mean_metrics.get("AUC-PR", 0)
            st.metric("AUC-PR", f"{auc_pr:.3f}")

    # Display per-class metrics first
    st.markdown("#### Per-Class Metrics")

    # Filter out the 'mean' column for per-class display
    per_class_results = (
        metrics_df.drop(columns=["mean"])
        if "mean" in metrics_df.columns
        else metrics_df
    )

    # Remove overall metrics rows (AUC-PR and mAP@[0.5:0.95]) from per-class display
    per_class_results = per_class_results.drop(
        ["AUC-PR", "mAP@[0.5:0.95]"], errors="ignore"
    )

    # Create a more readable display
    display_df = per_class_results.copy()

    # Round numeric columns for better display
    numeric_columns = display_df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(3)

    st.dataframe(display_df, width="stretch")

    # Now display Precision-Recall Curve
    if metrics_factory is not None:
        st.markdown("#### Precision-Recall Curve")

        try:
            # Get the precision-recall curve data
            curve_data = metrics_factory.get_overall_precision_recall_curve()
            auc_pr = metrics_factory.compute_auc_pr()

            # Create the plot using streamlit's plotly integration
            import plotly.graph_objects as go

            # Create the precision-recall curve
            fig = go.Figure()

            # Add the curve
            fig.add_trace(
                go.Scatter(
                    x=curve_data["recall"],
                    y=curve_data["precision"],
                    mode="lines",
                    name="Precision-Recall Curve",
                    line=dict(color="blue", width=2),
                    fill="tonexty",
                    fillcolor="rgba(0, 0, 255, 0.1)",
                )
            )

            # Add AUC-PR annotation
            fig.add_annotation(
                x=0.6,
                y=0.2,
                text=f"AUC-PR: {auc_pr:.3f}",
                showarrow=False,
                font=dict(size=12),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
            )

            # Update layout
            fig.update_layout(
                # title='Overall Precision-Recall Curve',
                xaxis_title="Recall",
                yaxis_title="Precision",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                showlegend=True,
                height=500,
            )

            # Add grid
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

            st.plotly_chart(fig, width="stretch")

        except Exception as e:
            st.error(f"Error plotting precision-recall curve: {e}")
            st.info("Precision-recall curve data not available.")

    # Download results
    st.markdown("#### Download Results")

    # Convert to CSV for download
    csv = metrics_df.to_csv(index=True)
    st.download_button(
        label="📥 Download per class metrics",
        data=csv,
        file_name="evaluation_results.csv",
        mime="text/csv",
    )
    try:
        curve_data = (
            metrics_factory.get_overall_precision_recall_curve()
            if metrics_factory is not None
            else None
        )
        if curve_data is not None:
            import pandas as pd

            pr_points_df = pd.DataFrame(
                {"recall": curve_data["recall"], "precision": curve_data["precision"]}
            )
            pr_csv = pr_points_df.to_csv(index=False)
            st.download_button(
                label="📈 Download precision-recall points",
                data=pr_csv,
                file_name="precision_recall_points.csv",
                mime="text/csv",
            )
        else:
            st.write("No precision-recall data available.")
    except Exception as e:
        st.write(f"Error preparing precision-recall points: {e}")

    # Show detailed statistics
    with st.expander("📊 Detailed Statistics"):
        st.markdown("**Results Shape:**")
        st.write(f"Rows: {metrics_df.shape[0]}, Columns: {metrics_df.shape[1]}")

        st.markdown("**Available Metrics:**")
        st.write(list(metrics_df.columns))

        st.markdown("**Class Names:**")
        st.write(
            list(metrics_df.index) if len(metrics_df.index) > 0 else "No classes found"
        )

        st.markdown("**DataFrame Info:**")
        st.write("Index:", metrics_df.index.tolist())
        st.write("Columns:", metrics_df.columns.tolist())

        st.markdown("**Sample Data:**")
        st.dataframe(metrics_df.head(), width="stretch")

        if "evaluation_config" in st.session_state:
            st.markdown("**Evaluation Configuration:**")
            config = st.session_state["evaluation_config"]
            for key, value in config.items():
                st.write(f"- {key}: {value}")

        # Show precision-recall curve data if available
        if metrics_factory is not None:
            st.markdown("**Precision-Recall Curve Data:**")
            try:
                curve_data = metrics_factory.get_overall_precision_recall_curve()
                st.write(f"Number of points: {len(curve_data['precision'])}")
                st.write(
                    f"Precision range: {min(curve_data['precision']):.3f} - {max(curve_data['precision']):.3f}"
                )
                st.write(
                    f"Recall range: {min(curve_data['recall']):.3f} - {max(curve_data['recall']):.3f}"
                )
                st.write(f"AUC-PR: {metrics_factory.compute_auc_pr():.3f}")
            except Exception as e:
                st.write(f"Error accessing curve data: {e}")
