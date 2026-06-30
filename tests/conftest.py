import random

import numpy as np
import pytest


SEED = 0


@pytest.fixture(autouse=True)
def global_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
