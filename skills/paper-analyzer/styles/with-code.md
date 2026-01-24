# with-code

结合 GitHub 开源代码进行讲解

## 使用方式

- 克隆或浏览论文的 GitHub 仓库
- 找到核心实现代码
- 贴出关键代码片段并讲解
- 将论文概念与代码实现对应

## 示例

```markdown
论文中的门控机制在代码中是这样实现的：

```python
# engram/model.py
def compute_gate(self, hidden_state, memory_key):
    # 计算隐藏状态和记忆 key 的对齐分数
    score = torch.matmul(hidden_state, memory_key.T)
    gate = torch.sigmoid(score / self.temperature)
    return gate
```

可以看到，门控值就是隐藏状态和记忆 key 的点积，
经过 sigmoid 归一化到 0-1 之间。
```

## 适用场景

想要复现或深入理解实现细节的读者
