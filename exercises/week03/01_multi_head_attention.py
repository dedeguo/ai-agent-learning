from block_utils import X, multi_head, show
import numpy as np

out, weights, concat = multi_head(X)
show("X", X)
for i, a in enumerate(weights):
    show(f"head {i+1} weights", a)
    np.testing.assert_allclose(a.sum(axis=-1), 1.)
show("concat", concat)
show("output", out)
assert out.shape == X.shape
