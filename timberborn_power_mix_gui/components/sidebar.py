import streamlit as st
from timberborn_power_mix.simulation import consts as sim_consts
from timberborn_power_mix.machines import FactoryName, ProducerName
from timberborn_power_mix.structures import CommonConfigName


def render_sidebar():
    """Renders the sidebar configuration and returns the configuration values."""

    # --- Sidebar: Configuration ---
    with st.sidebar.expander("Common Configuration", expanded=True):
        days = st.number_input(
            "Simulation Days", value=sim_consts.DEFAULT_DAYS, min_value=1, max_value=365
        )
        working_hours = st.number_input(
            "Working Hours",
            value=sim_consts.DEFAULT_WORKING_HOURS,
            min_value=0,
            max_value=24,
        )
        wet_days = st.number_input(
            "Wet Season Days",
            value=sim_consts.DEFAULT_WET_SEASON_DAYS,
            min_value=1,
            max_value=100,
        )
        dry_days = st.number_input(
            "Dry Season Days",
            value=sim_consts.DEFAULT_DRY_SEASON_DAYS,
            min_value=1,
            max_value=100,
        )
        badtide_days = st.number_input(
            "Badtide Season Days",
            value=sim_consts.DEFAULT_BADTIDE_SEASON_DAYS,
            min_value=1,
            max_value=100,
        )

    # Determine if we should expand the sections (e.g. if inputs were just updated)
    # We can use a session state flag for this, or just check if values are non-zero.
    # However, since we want to show the user what happened *after* clicking load template,
    # we can check if 'template_loaded' is in session state.

    expand_factories = False
    expand_energy_mix = False

    if st.session_state.get("template_loaded"):
        expand_factories = True
        expand_energy_mix = True
        # Reset the flag so they don't stay permanently expanded if the user collapses them
        st.session_state.template_loaded = False

    factory_counts = {}
    with st.sidebar.expander("Factories", expanded=expand_factories):
        for name in FactoryName:
            display_name = name.replace("_", " ").title()
            key = f"input_factory_{name.value}"

            # Ensure key exists
            if key not in st.session_state:
                st.session_state[key] = st.session_state.factory_counts_state[
                    name.value
                ]

            val = st.number_input(display_name, min_value=0, max_value=1000, key=key)
            st.session_state.factory_counts_state[name.value] = val
            factory_counts[name.value] = val

    producer_counts = {}
    with st.sidebar.expander("Energy Mix (Simulation)", expanded=expand_energy_mix):
        for name in ProducerName:
            display_name = name.replace("_", " ").title()
            key = f"input_{name.value}"

            # Ensure the key is in session state (for first run)
            if key not in st.session_state:
                st.session_state[key] = st.session_state.energy_mix_state[name.value]

            # Use session state to control the value
            val = st.number_input(
                display_name,
                min_value=0,
                max_value=1000,
                key=key,
            )
            # Update session state when user changes input
            st.session_state.energy_mix_state[name.value] = val
            producer_counts[name.value] = val

        key_batt = "input_battery_str"
        if key_batt not in st.session_state:
            st.session_state[key_batt] = st.session_state.energy_mix_state[
                "battery_str"
            ]

        battery_str = st.text_input(
            "Battery Heights (comma separated)",
            key=key_batt,
        )
        st.session_state.energy_mix_state["battery_str"] = battery_str

    battery_heights = [
        int(h.strip()) for h in battery_str.split(",") if h.strip().isdigit()
    ]

    common_params = {
        CommonConfigName.DAYS: days,
        CommonConfigName.WORKING_HOURS: working_hours,
        CommonConfigName.WET_DAYS: wet_days,
        CommonConfigName.DRY_DAYS: dry_days,
        CommonConfigName.BADTIDE_DAYS: badtide_days,
    }

    return common_params, factory_counts, producer_counts, battery_heights
