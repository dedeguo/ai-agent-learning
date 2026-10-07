from block_utils import X, positions, block, show
import numpy as np

result = block(X + positions(*X.shape))
for name, value in result.items():
    show(name, value)
assert result["Y"].shape == X.shape
assert np.isfinite(result["Y"]).all()
