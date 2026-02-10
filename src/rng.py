import os
from numpy.random import Generator, PCG64

_RNG = None
_RNG_PID = None


def get_rng():
    global _RNG, _RNG_PID
    current_pid = os.getpid()

    # If RNG is not initialized, or if we are in a different process than where it was initialized
    # (which happens after a fork), we need to re-initialize it.
    if _RNG is None or _RNG_PID != current_pid:
        _RNG = Generator(PCG64())
        _RNG_PID = current_pid

    return _RNG


def seed_rng(seed):
    global _RNG, _RNG_PID
    _RNG = Generator(PCG64(seed))
    _RNG_PID = os.getpid()
