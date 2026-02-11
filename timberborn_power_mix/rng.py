from multiprocessing.managers import BaseManager
from numpy.random import Generator, PCG64


class RNGService:
    """
    A service that provides independent, seeded random number generators.
    This service is intended to be run by a manager process, and accessed
    via a proxy.
    """

    def __init__(self, seed=None):
        # The root source of entropy/streams
        self._main_rng = PCG64(seed)

    def get_generator(self) -> Generator:
        """
        Spawns a new, independent Generator instance from the main seed.
        This method is thread-safe.
        """
        # spawn(1) returns a list of one new PCG64 instance
        new_stream = self._main_rng.spawn(1)[0]

        # Note: When returning via multiprocessing Manager, the object is pickled.
        # This means the caller gets a COPY of the generator state.
        return Generator(new_stream)


class RNGManager(BaseManager):
    """A manager to host and expose the RNGService."""

    pass


# Register the service with the manager.
RNGManager.register("RNGService", RNGService)
