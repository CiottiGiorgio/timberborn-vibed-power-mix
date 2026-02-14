from typing import NamedTuple
import numpy as np


class SimulationSample(NamedTuple):
    power_production: np.ndarray
    battery_charge: np.ndarray


class AggregatedSamples(NamedTuple):
    hours_empty_results: np.ndarray
    final_surpluses: np.ndarray
    power_consumption: np.ndarray


class BatchedSimulationResult(NamedTuple):
    worst_sample: SimulationSample
    aggregated_samples: AggregatedSamples
