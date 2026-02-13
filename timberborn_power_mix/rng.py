from numpy.random import Generator, PCG64


class RNGService:
    """
    A service that provides independent, seeded random number generators.
    """

    def __init__(self, seed=None):
        # The root source of entropy/streams
        self._main_rng = PCG64(seed)

    def get_generator(self) -> Generator:
        """
        Spawns a new, independent Generator instance from the main seed.
        """
        # spawn(1) returns a list of one new PCG64 instance
        new_stream = self._main_rng.spawn(1)[0]
        return Generator(new_stream)
