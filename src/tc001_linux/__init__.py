"""TC001 Mini thermal viewer for Linux."""

import numpy as np
from numpy.typing import NDArray

type Plane = NDArray[np.float32]
type Image = NDArray[np.uint8]
type Color = tuple[int, int, int]
