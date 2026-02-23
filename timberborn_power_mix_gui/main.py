import streamlit as st

from timberborn_power_mix.machines import (
    FactoryName,
    ProducerName,
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

    st.markdown("""
    Welcome to the **Timberborn Vibed Power Mix** tool! This application helps you design the perfect power grid for your beaver colony.
    
    ### How to use:
    1. **Configure Seasons**: Set the duration of Wet, Dry, and Badtide seasons in the sidebar.
    2. **Add Factories**: Specify how many of each factory type you want to power.
    3. **Choose an Action**:
        - **🚀 Run Simulation**: Test a specific mix of power producers (Windmills, Water Wheels, etc.) and batteries to see how they perform.
        - **🎯 Run Optimization**: Let the AI find the most efficient (cheapest) mix of power sources and batteries to keep your factories running reliably.
    """)

    # --- Sidebar: Configuration ---
    st.sidebar.header("Common Configuration")
    days = st.sidebar.number_input(
        "Simulation Days", value=sim_consts.DEFAULT_DAYS, min_value=1, max_value=365
    )
    working_hours = st.sidebar.number_input(
        "Working Hours",
        value=sim_consts.DEFAULT_WORKING_HOURS,
        min_value=0,
        max_value=24,
    )
    wet_days = st.sidebar.number_input(
        "Wet Season Days",
        value=sim_consts.DEFAULT_WET_SEASON_DAYS,
        min_value=1,
        max_value=100,
    )
    dry_days = st.sidebar.number_input(
        "Dry Season Days",
        value=sim_consts.DEFAULT_DRY_SEASON_DAYS,
        min_value=1,
        max_value=100,
    )
    badtide_days = st.sidebar.number_input(
        "Badtide Season Days",
        value=sim_consts.DEFAULT_BADTIDE_SEASON_DAYS,
        min_value=1,
        max_value=100,
    )

    # Hidden/Fixed parameters
    samples = sim_consts.DEFAULT_SIMULATION_SAMPLES
    max_time = opt_consts.DEFAULT_MAX_TIME_SECONDS
    target_unreliability = opt_consts.DEFAULT_TARGET_UNRELIABILITY

    st.sidebar.header("Factories")
    factory_counts = {}
    for name in FactoryName:
        display_name = name.replace("_", " ").title()
        factory_counts[name.value] = st.sidebar.number_input(
            display_name, value=0, min_value=0, max_value=1000
        )

    st.sidebar.header("Energy Mix (Simulation)")
    producer_counts = {}
    for name in ProducerName:
        display_name = name.replace("_", " ").title()
        producer_counts[name.value] = st.sidebar.number_input(
            display_name, value=0, min_value=0, max_value=1000
        )

    battery_str = st.sidebar.text_input("Battery Heights (comma separated)", value="")
    battery_heights = [
        int(h.strip()) for h in battery_str.split(",") if h.strip().isdigit()
    ]

    # --- Main Area: Actions and Results ---
    col1, col2 = st.columns(2)

    run_sim = col1.button("🚀 Run Simulation", use_container_width=True)
    run_opt = col2.button("🎯 Run Optimization", use_container_width=True)

    # State management for results
    if "sim_result" not in st.session_state:
        st.session_state.sim_result = None
    if "sim_config" not in st.session_state:
        st.session_state.sim_config = None

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

            st.success(f"Optimization Finished! Best wood cost: {opt_res.best_cost}")
            st.info(f"Optimized Mix: {opt_res.best_mix}")

    if st.session_state.sim_result and st.session_state.sim_config:
        fig = create_simulation_figure(
            st.session_state.sim_config, st.session_state.sim_result
        )
        st.pyplot(fig)


if __name__ == "__main__":
    main()
