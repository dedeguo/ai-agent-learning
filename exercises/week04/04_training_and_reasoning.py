"""预先给定候选 + 精确 verifier，演示推理时选择，不调用模型。"""
candidates = ['17', '18', '十八', '无法判断']
expected = 7 + 11


def verify(answer):
    try:
        return int(answer.strip()) == expected
    except ValueError:
        return False


for answer in candidates:
    print(f'问题：7 + 11 = ? 候选：{answer!r} 检查通过：{verify(answer)}')
selected = next((answer for answer in candidates if verify(answer)), None)
print('选中的候选：', selected)
assert selected == '18'
print('没有参数更新；这不是 SFT、RLHF 或 DPO。')
print('本检查器只接受整数格式，语义正确的“十八”也会被拒绝，说明验收规则同样影响结果。')
