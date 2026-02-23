import streamlit as st
from timberborn_power_mix.machines import FactoryName, ProducerName


def render_template_button(parent):
    """Renders the template button in the given parent container and updates state if clicked."""

    # Show toast if flag is set (from previous run)
    if st.session_state.get("show_template_toast"):
        st.toast("Example template loaded!", icon="✅")
        st.session_state.show_template_toast = False

    if parent.button("📋 Load Example", use_container_width=True):
        st.session_state.factory_counts_state = {name.value: 0 for name in FactoryName}
        st.session_state.factory_counts_state[FactoryName.LUMBER_MILLS.value] = 3
        st.session_state.factory_counts_state[FactoryName.GEAR_WORKSHOPS.value] = 2
        st.session_state.factory_counts_state[FactoryName.STEEL_FACTORIES.value] = 3
        st.session_state.factory_counts_state[FactoryName.PAPER_MILLS.value] = 3
        st.session_state.factory_counts_state[FactoryName.PRINTING_PRESSES.value] = 2
        st.session_state.factory_counts_state[FactoryName.WOOD_WORKSHOPS.value] = 2
        st.session_state.factory_counts_state[FactoryName.GRILLMISTS.value] = 1
        st.session_state.factory_counts_state[FactoryName.CENTRIFUGES.value] = 1
        st.session_state.factory_counts_state[
            FactoryName.EXPLOSIVES_FACTORIES.value
        ] = 1
        st.session_state.factory_counts_state[FactoryName.BOT_PART_FACTORIES.value] = 3
        st.session_state.factory_counts_state[FactoryName.BOT_ASSEMBLERS.value] = 1

        st.session_state.energy_mix_state = {name.value: 0 for name in ProducerName}
        st.session_state.energy_mix_state[ProducerName.WINDMILLS.value] = 3
        st.session_state.energy_mix_state[ProducerName.LARGE_WINDMILLS.value] = 15
        st.session_state.energy_mix_state["battery_str"] = "45"

        st.session_state.update_inputs = True
        st.session_state.template_loaded = True
        st.session_state.show_template_toast = True
        st.rerun()
