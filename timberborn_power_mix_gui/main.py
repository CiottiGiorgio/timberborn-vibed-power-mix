import streamlit as st

from timberborn_power_mix.machines import (
    FactoryName,
    ProducerName,
    BatteryName,
)
from timberborn_power_mix.simulation import consts as sim_consts
from timberborn_power_mix.optimization import consts as opt_consts
from timberborn_power_mix.simulation.models import SimulationConfig, EnergyMixConfig
from timberborn_power_mix.optimization.models import OptimizationConfig
from timberborn_power_mix.models import FactoryConfig
from timberborn_power_mix.structures import (
    CommonConfigName,
    OptimizeConfigName,
    Percentile,
)
from timberborn_power_mix.simulation.orchestrator import run_simulation
from timberborn_power_mix.optimization.engine import run_optimization
from timberborn_power_mix.plots.canvas import create_simulation_figure


def main():
    st.set_page_config(page_title="Timberborn Vibed Power Mix", layout="wide")
    st.title("🦫 Timberborn Vibed Power Mix")

    st.markdown(f"""
    Welcome to the **Timberborn Vibed Power Mix** tool! This application helps you design the perfect power grid for your beaver colony.
    
    ### How to use:
    1. **Configure Seasons**: Set the duration of Wet, Dry, and Badtide seasons in the sidebar.
    2. **Add Factories**: Specify how many of each factory type you want to power.
    3. **Choose an Action**:
        - **🚀 Run Simulation**: Test a specific mix of power producers (Windmills, Water Wheels, etc.) and batteries to see how they perform.
        - **🎯 Run Optimization**: Let the AI find the most efficient (cheapest) mix of power sources and batteries to keep your factories running reliably.
    
    ### Disclaimers:
    - **Faction**: This tool assumes you are playing as the **Folktails** faction.
    - **Costs**: All building costs (planks, gears, metal blocks, etc.) are converted into an equivalent **raw wood cost** for comparison.
    - **Optimization**: The optimization process runs for **{opt_consts.DEFAULT_MAX_TIME_SECONDS} seconds** and returns the best solution found within that time limit.
    - **Overwrite**: Running an optimization will overwrite your current Energy Mix configuration with the optimal result found.
    """)

    # Initialize session state for energy mix if not present
    if "energy_mix_state" not in st.session_state:
        st.session_state.energy_mix_state = {name.value: 0 for name in ProducerName}
        st.session_state.energy_mix_state["battery_str"] = ""

    # Initialize session state for factory counts if not present
    if "factory_counts_state" not in st.session_state:
        st.session_state.factory_counts_state = {name.value: 0 for name in FactoryName}

    # Check if we need to update inputs from a previous optimization run or template load
    if st.session_state.get("update_inputs"):
        for name in ProducerName:
            st.session_state[f"input_{name.value}"] = st.session_state.energy_mix_state[
                name.value
            ]
        st.session_state["input_battery_str"] = st.session_state.energy_mix_state[
            "battery_str"
        ]

        # Also update factory inputs
        for name in FactoryName:
            st.session_state[f"input_factory_{name.value}"] = (
                st.session_state.factory_counts_state[name.value]
            )

        st.session_state.update_inputs = False

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

    # Hidden/Fixed parameters
    samples = sim_consts.DEFAULT_SIMULATION_SAMPLES
    max_time = opt_consts.DEFAULT_MAX_TIME_SECONDS
    target_unreliability = opt_consts.DEFAULT_TARGET_UNRELIABILITY

    factory_counts = {}
    with st.sidebar.expander("Factories", expanded=False):
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
    with st.sidebar.expander("Energy Mix (Simulation)", expanded=False):
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

    # --- Main Area: Actions and Results ---

    # Templates Section
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

    st.markdown("---")
    st.markdown("### ⚙️ Actions")

    col1, col2 = st.columns(2)

    run_sim = col1.button("🚀 Run Simulation", use_container_width=True)
    run_opt = col2.button("🎯 Run Optimization", use_container_width=True)

    # State management for results
    if "sim_result" not in st.session_state:
        st.session_state.sim_result = None
    if "sim_config" not in st.session_state:
        st.session_state.sim_config = None
    if "optimization_success" not in st.session_state:
        st.session_state.optimization_success = None

    common_params = {
        CommonConfigName.DAYS: days,
        CommonConfigName.WORKING_HOURS: working_hours,
        CommonConfigName.WET_DAYS: wet_days,
        CommonConfigName.DRY_DAYS: dry_days,
        CommonConfigName.BADTIDE_DAYS: badtide_days,
        CommonConfigName.SAMPLES: samples,
        CommonConfigName.SEED: None,
        CommonConfigName.THREADS: None,
    }
    factories = FactoryConfig(**factory_counts)

    if run_sim:
        with st.spinner("Running simulation..."):
            energy_mix = EnergyMixConfig(
                battery_heights=battery_heights, **producer_counts
            )
            config = SimulationConfig(
                **common_params, factories=factories, energy_mix=energy_mix
            )
            res = run_simulation(config)
            st.session_state.sim_result = res
            st.session_state.sim_config = config
            st.session_state.optimization_success = None  # Clear previous opt message

    if run_opt:
        with st.spinner("Running optimization..."):
            opt_common = common_params.copy()
            opt_common[CommonConfigName.SAMPLES] = (
                opt_consts.DEFAULT_OPTIMIZATION_SAMPLES
            )

            opt_config = OptimizationConfig(
                **opt_common,
                factories=factories,
                max_time=max_time,
                target_unreliability=target_unreliability,
                percentile=Percentile.P95,
            )

            opt_res = run_optimization(opt_config)

            # Update session state with optimized values
            for name in ProducerName:
                val = getattr(opt_res.best_mix, name.value)
                st.session_state.energy_mix_state[name.value] = val

            # Update battery string
            heights = getattr(opt_res.best_mix, BatteryName.BATTERY_HEIGHTS.value)
            battery_str = ", ".join(map(str, heights))
            st.session_state.energy_mix_state["battery_str"] = battery_str

            # Simulate the best result
            sim_data = opt_config.model_dump()
            sim_data.pop(OptimizeConfigName.MAX_TIME)
            sim_data.pop(OptimizeConfigName.TARGET_UNRELIABILITY)
            sim_data.pop(OptimizeConfigName.PERCENTILE)
            sim_data[CommonConfigName.SAMPLES] = sim_consts.DEFAULT_SIMULATION_SAMPLES

            sim_config = SimulationConfig(**sim_data, energy_mix=opt_res.best_mix)
            sim_res = run_simulation(sim_config)

            st.session_state.sim_result = sim_res
            st.session_state.sim_config = sim_config

            # Store success message in session state to persist after rerun
            st.session_state.optimization_success = {
                "cost": opt_res.best_cost,
                "mix": opt_res.best_mix,
            }

            # Flag to update inputs on next run
            st.session_state.update_inputs = True

            # Force a rerun to update the sidebar inputs
            st.rerun()

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


if __name__ == "__main__":
    main()
