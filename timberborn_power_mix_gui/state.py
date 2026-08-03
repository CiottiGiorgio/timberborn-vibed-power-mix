import streamlit as st

from timberborn_power_mix.machines import FactoryName, ProducerName


def init_session_state():
    """Initializes the session state variables if they don't exist."""
    if "energy_mix_state" not in st.session_state:
        st.session_state.energy_mix_state = {name.value: 0 for name in ProducerName}
        st.session_state.energy_mix_state["battery_str"] = ""

    if "factory_counts_state" not in st.session_state:
        st.session_state.factory_counts_state = {name.value: 0 for name in FactoryName}

    if "sim_result" not in st.session_state:
        st.session_state.sim_result = None
    if "sim_config" not in st.session_state:
        st.session_state.sim_config = None
    if "optimization_success" not in st.session_state:
        st.session_state.optimization_success = None


def update_inputs_from_state():
    """Updates the widget input keys from the session state if the update flag is set."""
    if st.session_state.get("update_inputs"):
        for name in ProducerName:
            st.session_state[f"input_{name.value}"] = st.session_state.energy_mix_state[
                name.value
            ]
        st.session_state["input_battery_str"] = st.session_state.energy_mix_state[
            "battery_str"
        ]

        for name in FactoryName:
            st.session_state[f"input_factory_{name.value}"] = (
                st.session_state.factory_counts_state[name.value]
            )

        st.session_state.update_inputs = False
