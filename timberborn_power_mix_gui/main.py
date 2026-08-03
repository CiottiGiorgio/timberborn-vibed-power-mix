import streamlit as st

from timberborn_power_mix.machines import (
    BatteryName,
    ProducerName,
)
from timberborn_power_mix.models import FactoryConfig
from timberborn_power_mix.optimization import consts as opt_consts
from timberborn_power_mix.optimization.engine import run_optimization
from timberborn_power_mix.optimization.models import OptimizationConfig
from timberborn_power_mix.simulation import consts as sim_consts
from timberborn_power_mix.simulation.models import EnergyMixConfig, SimulationConfig
from timberborn_power_mix.simulation.orchestrator import run_simulation
from timberborn_power_mix.structures import (
    CommonConfigName,
    OptimizeConfigName,
    Percentile,
)
from timberborn_power_mix_gui.components.results import render_results
from timberborn_power_mix_gui.components.sidebar import render_sidebar
from timberborn_power_mix_gui.components.templates import render_template_button
from timberborn_power_mix_gui.state import init_session_state, update_inputs_from_state


def main():
    st.set_page_config(page_title="Timberborn Vibed Power Mix", layout="wide")
    st.title("🦫 Timberborn Vibed Power Mix")

    st.markdown(f"""
    Welcome to the **Timberborn Vibed Power Mix** tool! This application helps you design the perfect power grid for your beaver colony.
    
    ### How to use:
    1. **Configure Seasons**: Set the duration of Wet, Dry, and Badtide seasons in the sidebar.
    2. **Add Factories**: Specify how many of each factory type you want to power.
    3. **Choose an Action**:
        - **📋 Load Example**: Load a pre-configured example to get started quickly.
        - **🚀 Run Simulation**: Test a specific mix of power producers (Windmills, Water Wheels, etc.) and batteries to see how they perform.
        - **🎯 Run Optimization**: Let the AI find the most efficient (cheapest) mix of power sources and batteries to keep your factories running reliably.
    
    ### Disclaimers:
    - **Faction**: This tool assumes you are playing as the **Folktails** faction.
    - **Costs**: All building costs (planks, gears, metal blocks, etc.) are converted into an equivalent **raw wood cost** for comparison.
    - **Optimization**: The optimization process runs for **{opt_consts.DEFAULT_MAX_TIME_SECONDS} seconds** and returns the best solution found within that time limit.
    - **Overwrite**: Running an optimization will overwrite your current Energy Mix configuration with the optimal result found.
    - **Reliability**: The tool optimizes for **productivity loss**, meaning it tries to minimize the percentage of potential work lost due to power shortages.
    """)

    # Initialize and update state
    init_session_state()
    update_inputs_from_state()

    # Render Sidebar and get configuration
    common_params_dict, factory_counts, producer_counts, battery_heights = (
        render_sidebar()
    )

    st.markdown("---")
    st.markdown("### ⚙️ Actions")

    # Create 3 columns for actions
    col1, col2, col3 = st.columns(3)

    # 1. Load Template
    render_template_button(col1)

    # 2. Run Simulation
    run_sim = col2.button("🚀 Run Simulation", use_container_width=True)

    # 3. Run Optimization
    run_opt = col3.button("🎯 Run Optimization", use_container_width=True)

    # Hidden/Fixed parameters
    samples = sim_consts.DEFAULT_SIMULATION_SAMPLES
    max_time = opt_consts.DEFAULT_MAX_TIME_SECONDS
    target_unreliability = opt_consts.DEFAULT_TARGET_UNRELIABILITY

    # Prepare common params
    common_params = {
        CommonConfigName.DAYS: common_params_dict[CommonConfigName.DAYS],
        CommonConfigName.WORKING_HOURS: common_params_dict[
            CommonConfigName.WORKING_HOURS
        ],
        CommonConfigName.WET_DAYS: common_params_dict[CommonConfigName.WET_DAYS],
        CommonConfigName.DRY_DAYS: common_params_dict[CommonConfigName.DRY_DAYS],
        CommonConfigName.BADTIDE_DAYS: common_params_dict[
            CommonConfigName.BADTIDE_DAYS
        ],
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
            st.session_state.show_opt_toast = True

            # Force a rerun to update the sidebar inputs
            st.rerun()

    # Render Results
    render_results()


if __name__ == "__main__":
    main()
