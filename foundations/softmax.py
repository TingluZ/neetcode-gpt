import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        
        n = np.exp(z - max(z))
        d = 0
        for i in z:
            d += np.exp(i - max(z))
        
        result = n / d

        return np.round(result, 4)