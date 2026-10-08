import numpy as np
from gpt_utils import VOCAB, gpt, softmax, cross_entropy

ids = np.array([0, 1, 2, 3, 4])
inputs, targets = ids[:-1], ids[1:]
logits = gpt(inputs)
losses = cross_entropy(logits, targets)
for i, (source, target) in enumerate(zip(inputs, targets)):
    print(f'{VOCAB[source]} → {VOCAB[target]}: 目标概率={softmax(logits)[i, target]:.4f}, loss={losses[i]:.4f}')
print('平均 loss:', losses.mean())
improved = logits.copy()
improved[np.arange(len(targets)), targets] += 2
new_losses = cross_entropy(improved, targets)
print('人为提高正确目标 logit 后的平均 loss:', new_losses.mean())
print('注意：本实验没有反向传播和参数更新。')
assert np.all(new_losses < losses)
assert np.allclose(cross_entropy(np.zeros_like(logits), targets), np.log(len(VOCAB)))
