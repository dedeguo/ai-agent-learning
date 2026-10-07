from block_utils import X, multi_head, positions, show
import numpy as np

order = [2, 1, 0]
p = positions(*X.shape)
a = multi_head(X[order])[0]
b = multi_head(X)[0][order]
np.testing.assert_allclose(a, b, atol=1e-12)
print("无位置编码，重排误差:", np.max(np.abs(a-b)))
a_pos = multi_head(X[order] + p)[0]
b_pos = multi_head(X + p)[0][order]
show("P", p)
print("有位置编码，重排误差:", np.max(np.abs(a_pos-b_pos)))
