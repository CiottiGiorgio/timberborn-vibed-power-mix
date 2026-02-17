import os
from typing import Optional


def calculate_optimal_threads(
    requested_threads: Optional[int], data_cardinality: int
) -> int:
    """
    Determines the number of threads to use based on the system, user request,
    and data cardinality.
    """
    cpu_count = os.process_cpu_count() or os.cpu_count() or 1
    base_threads = requested_threads or cpu_count
    return max(1, min(base_threads, data_cardinality))
