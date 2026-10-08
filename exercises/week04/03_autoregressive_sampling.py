import numpy as np
from gpt_utils import VOCAB, gpt, softmax, sample

fixed_logits = np.array([2., 1., 0.])
for temperature in (.3, 1., 2.):
    probs = softmax(fixed_logits / temperature)
    print(f'T={temperature}: {np.round(probs, 4)}')
    assert np.isclose(probs.sum(), 1)

for temperature in (0., 1.):
    rng = np.random.default_rng(42)
    context = [0, 1]
    print(f'\nT={temperature}, 提示:', ' '.join(VOCAB[i] for i in context))
    for step in range(8):
        logits = gpt(context)[-1]
        next_id = sample(logits, temperature, rng)
        context.append(next_id)
        print(f'第 {step+1} 步追加 {VOCAB[next_id]}:', ' '.join(VOCAB[i] for i in context))
        if next_id == VOCAB.index('<eos>'):
            print('遇到结束 token，停止。')
            break
    else:
        print('达到最大生成长度，停止。')
print('\n随机模型尚未训练，文本不连贯属于预期现象。')
