from typing import List, Optional
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


# define unigram model
# credit to Patrick


class UnigramModel:
    """The unigram language model."""

    def __init__(self, size: int) -> None:
        """Initialize."""
        self.size = size
        self.p: Optional[FloatArray] = None

    def train(self, encodings: List[FloatArray]) -> "UnigramModel":
        """Train the model on data."""
        counts = np.ones((self.size, 1))
        for encoding in encodings:
            counts += encoding
        self.p = counts / counts.sum()
        return self

    def apply(self, encodings: List[FloatArray]) -> float:
        """Compute the log probability of a document."""
        if self.p is None:
            raise ValueError("This model is untrained")
        return (
            np.hstack(encodings).sum(axis=1, keepdims=True).T @ np.log(self.p)
        ).item()
