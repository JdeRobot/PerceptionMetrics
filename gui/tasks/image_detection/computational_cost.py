import streamlit as st
import pandas as pd


def render_computational_cost():
    """Render the computational cost tab."""

    st.header("Computational Cost")
    st.markdown(
        "Analyze the computational cost of your loaded image detection model."
    )

    # Check whether a model is loaded
    model = st.session_state.get("detection_model")

    if model is None:
        st.warning(
            "No model loaded. Please load a model using the sidebar first."
        )
        return

    # Benchmark configuration
    st.markdown("### Benchmark Configuration")

    col1, col2 = st.columns(2)

    with col1:
        height = st.number_input(
            "Input Height",
            min_value=32,
            value=224,
            step=32,
        )

    with col2:
        width = st.number_input(
            "Input Width",
            min_value=32,
            value=224,
            step=32,
        )

    runs = st.number_input(
        "Number of Runs",
        min_value=1,
        value=30,
        step=1,
    )

    warm_up_runs = st.number_input(
        "Warm-up Runs",
        min_value=0,
        value=5,
        step=1,
    )

    # Run computational cost analysis
    if st.button("Run Analysis", type="primary"):
        with st.spinner("Analyzing computational cost..."):
            try:
                results = model.get_computational_cost(
                    image_size=(height, width),
                    runs=runs,
                    warm_up_runs=warm_up_runs,
                )

                st.session_state["computational_cost_results"] = results

            except Exception as e:
                st.error(f"Computational cost analysis failed: {e}")

    # Display previously calculated results
    results = st.session_state.get("computational_cost_results")

    if results is not None:
        if not results.empty:
            row = results.iloc[0]

            inference_time = float(row["inference_time_s"])
            fps = 1 / inference_time if inference_time > 0 else 0

            st.markdown("### Computational Cost Results")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Parameters (M)", f"{row['n_params'] / 1_000_000:.2f}")
            col2.metric("Model Size (MB)", f"{row['size_mb']:.2f}" if pd.notna(row["size_mb"]) else "N/A")
            col3.metric("Inference Latency (ms)", f"{inference_time * 1000:.2f}")
            col4.metric("FPS", f"{fps:.2f}")

            st.markdown("### Complete Results")
            st.dataframe(results, use_container_width=True)

            st.download_button(
                label="Download Results as CSV",
                data=results.to_csv(index=False),
                file_name="computational_cost_results.csv",
                mime="text/csv",
            )
        else:
            st.warning("No computational cost results were returned.")
