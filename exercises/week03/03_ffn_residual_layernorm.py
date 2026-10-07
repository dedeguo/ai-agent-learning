from block_utils import X, layer_norm, ffn, show
import numpy as np

normalized = layer_norm(X)
show("LayerNorm(X)", normalized)
print("每行均值:", normalized.mean(axis=-1))
print("每行方差:", normalized.var(axis=-1))
show("FFN(X)", ffn(X))
show("X + FFN(X)", X + ffn(X))
changed = X.copy()
changed[0] += [1., -1., 2., 0.]
np.testing.assert_allclose(ffn(changed)[1:], ffn(X)[1:])
print("只修改第一行：FFN 其他行保持不变")
