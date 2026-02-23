import streamlit as st
from timberborn_power_mix.machines import FactoryName, ProducerName


def render_templates():
    """Renders the template buttons and updates the session state if clicked."""
    st.markdown("### 📋 Load Template")
    col_t1, col_t2, col_t3 = st.columns(3)

    if col_t1.button("Simple", use_container_width=True):
        # simulate-simple: --lumber-mills 1 --wood-workshops 1 --large-windmills 1 --windmills 1 --battery 5
        st.session_state.factory_counts_state = {name.value: 0 for name in FactoryName}
        st.session_state.factory_counts_state[FactoryName.LUMBER_MILLS.value] = 1
        st.session_state.factory_counts_state[FactoryName.WOOD_WORKSHOPS.value] = 1

        st.session_state.energy_mix_state = {name.value: 0 for name in ProducerName}
        st.session_state.energy_mix_state[ProducerName.LARGE_WINDMILLS.value] = 1
        st.session_state.energy_mix_state[ProducerName.WINDMILLS.value] = 1
        st.session_state.energy_mix_state["battery_str"] = "5"

        st.session_state.update_inputs = True
        st.rerun()

    if col_t2.button("Bot", use_container_width=True):
        # simulate-bot: --working-hours 24 --bot-part-factories 3 --bot-assemblers 1 --water-wheels 1 --large-windmills 5 --battery 13
        st.session_state.factory_counts_state = {name.value: 0 for name in FactoryName}
        st.session_state.factory_counts_state[FactoryName.BOT_PART_FACTORIES.value] = 3
        st.session_state.factory_counts_state[FactoryName.BOT_ASSEMBLERS.value] = 1

        st.session_state.energy_mix_state = {name.value: 0 for name in ProducerName}
        st.session_state.energy_mix_state[ProducerName.WATER_WHEELS.value] = 1
        st.session_state.energy_mix_state[ProducerName.LARGE_WINDMILLS.value] = 5
        st.session_state.energy_mix_state["battery_str"] = "13"

        st.session_state.update_inputs = True
        st.rerun()

    if col_t3.button("Complex", use_container_width=True):
        # simulate-complex: --working-hours 24 --lumber-mills 2 --gear-workshops 1 --steel-factories 2 --printing-presses 2 --wood-workshops 1 --grillmists 1 --centrifuges 1 --large-windmills 12 --battery 22
        st.session_state.factory_counts_state = {name.value: 0 for name in FactoryName}
        st.session_state.factory_counts_state[FactoryName.LUMBER_MILLS.value] = 2
        st.session_state.factory_counts_state[FactoryName.GEAR_WORKSHOPS.value] = 1
        st.session_state.factory_counts_state[FactoryName.STEEL_FACTORIES.value] = 2
        st.session_state.factory_counts_state[FactoryName.PRINTING_PRESSES.value] = 2
        st.session_state.factory_counts_state[FactoryName.WOOD_WORKSHOPS.value] = 1
        st.session_state.factory_counts_state[FactoryName.GRILLMISTS.value] = 1
        st.session_state.factory_counts_state[FactoryName.CENTRIFUGES.value] = 1

        st.session_state.energy_mix_state = {name.value: 0 for name in ProducerName}
        st.session_state.energy_mix_state[ProducerName.LARGE_WINDMILLS.value] = 12
        st.session_state.energy_mix_state["battery_str"] = "22"

        st.session_state.update_inputs = True
        st.rerun()
