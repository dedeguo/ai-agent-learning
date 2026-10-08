import numpy as np
from gpt_utils import attention, gpt

x = np.array([[1., 0., 0., 1.], [0., 1., 0., 1.], [1., 1., 0., 0.]])
changed = x.copy()
changed[-1] = [0., 0., 3., -2.]
for causal in (False, True):
    before, weights = attention(x, causal)
    after, _ = attention(changed, causal)
    print('\ncausal =', causal)
    print('第一个头的权重:\n', np.round(weights[0], 4))
    print('前两个位置输出的最大变化:', np.max(np.abs(after[:2] - before[:2])))
    assert np.allclose(weights.sum(axis=-1), 1)
    if causal:
        assert np.allclose(before[:2], after[:2])
        assert np.all(weights[:, np.triu_indices(3, 1)[0], np.triu_indices(3, 1)[1]] == 0)
    else:
        assert not np.allclose(before[:2], after[:2])
logits = gpt([0, 1, 2])
print('\n三个位置的词表 logits shape:', logits.shape)
assert logits.shape == (3, 6)
# 完整堆叠后，修改未来输入仍不能影响前面位置。
assert np.allclose(logits[:2], gpt([0, 1, 5])[:2])
