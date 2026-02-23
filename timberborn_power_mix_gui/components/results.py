import streamlit as st
from timberborn_power_mix.plots.canvas import create_simulation_figure


def render_results():
    """Renders the simulation results and optimization success message."""

    # Show toast if flag is set (from previous run)
    if st.session_state.get("show_opt_toast"):
        st.toast("Optimization complete!", icon="🎉")
        st.session_state.show_opt_toast = False

    # Display success message if it exists in session state
    if st.session_state.optimization_success:
        cost = st.session_state.optimization_success["cost"]
        mix = st.session_state.optimization_success["mix"]
        st.success(f"Optimization Finished! Best wood cost: {cost}")
        st.info(f"Optimized Mix: {mix}")

    if st.session_state.sim_result and st.session_state.sim_config:
        fig = create_simulation_figure(
            st.session_state.sim_config, st.session_state.sim_result
        )
        st.pyplot(fig)
