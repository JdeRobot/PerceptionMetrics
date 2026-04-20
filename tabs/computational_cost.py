"""Computational cost tab for the PerceptionMetrics GUI.

Wraps ``TorchImageDetectionModel.get_computational_cost`` in a Streamlit
tab so users can inspect parameter count, model size, and empirical
inference latency for the currently loaded detection model.
"""

import streamlit as st


def computational_cost_tab():
    """Render the Computational Cost tab.

    Requires a detection model in ``st.session_state.detection_model``.
    Configures input size and timing parameters, triggers the cost
    estimation on click, and renders the resulting metrics plus the
    full DataFrame.
    """

    st.header("Computatioal Csot")


    if not st.session_state.get("detetction_model_loaded",False):
        st.warning(
            "No detection model loaded. Use the sidebar to load a model first."
        )
        return
    
    model=st.session.state.detection_model()

    st.subheader("Input Configuration")
    col_h , col_w = st.columns(2)

    with col_h:
        height = st.number_input(
            "Image Height", min_value=32, max_value=4096, value=640, step=32,
        )
    with col_w:
        width = st.number_input(
            "Image width", min_value=32, max_value=4096, value=640, step=32,
            help="Dummy input width (pixels) passed to get_computational_cost.",
        )

        st.subheader("Timing Configuration")
    col_runs, col_warmup = st.columns(2)

    with col_runs:
        runs = st.number_input(
            "Timed runs", min_value=1, max_value=500, value=30, step=1,
            help="Number of forward passes timed and averaged.",
        )
    with col_warmup:
        warm_up_runs = st.number_input(
            "Warm-up runs", min_value=0, max_value=100, value=5, step=1,
            help="Forward passes before timing starts (stabilises GPU).",
        )


    if st.button("Run Cost Analysis", type= "primary"):
        with st.spinner("Running forward passses - do not close this tab..."):
            cost_df = model.get_computational_cost(
                image_size = (int(height), int(width)),
                runs = int(runs),
                warm_up_runs = int(warm_up_runs)
            )

        st.session_state.computational_cost_result = cost_df

    result_df = st.session_state.get("computational_cost_result")
    if result_df is None : 
        st.info("Configure inputs above and click **Run Cost Analysis**.")
        return
    
    print(result_df.iloc[0].to_dict())

    result = result_df.iloc[0].to_dict()

    st.subheader("Result")
    m1 ,m2 ,m3 , m4 = st.columns(4)
    m1.metric("Input shape" , result["input_shape"])
    m2.metric("Parameters", f"{result['n_params'] / 1e6:.2f} M")
    m3.metric(
        "Model size",
        f"{result['size_mb']:.2f} MB" if result["size_mb"] is not None else "N/A",
    )
    m4.metric(
        "Inference latency",
        f"{result['inference_time_s']* 1000:.2f} ms",
        help= f"Mean over {int(runs)} timed runs after {int(warm_up_runs)} warm-ups",
    )

    fps= 1.0 / result["inference_time_s"] if result["inference_time_s"] > 0 else 0.0
    st.metric("Throughput", f"{fps:.1f} FPS")


    with st.expander("Full raw Dataframe"):
        st.dataframe(result_df, use_container_width=True)
        st.download_button(
            label= "Download as CSV",
            data = result_df.to_csv(index=False).encode("utf-8"),
            file_name= "computational_cost.csv",
            mime="text/csv"
        )
    