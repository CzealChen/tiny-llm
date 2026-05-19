# LLM 推理完整流程

以 **Qwen2-0.5B** 作为具体模型，示例 prompt 为 `"你好"`，从字符串输入到字符串输出，全程追踪每一步的维度变化和数据变换。

---

## 模型参数

```
Qwen2-0.5B:
  vocab_size          = 151936
  hidden_size         = 896
  num_attention_heads = 14
  num_kv_heads        = 2
  head_dim            = 896 / 14 = 64
  intermediate_size   = 4864
  num_hidden_layers   = 24
  GQA 分组数          = 14 / 2 = 7 (7个query头共享1对kv头)
  tie_word_embeddings = True

推理精度: float16
```

---

## 总览：端到端数据流

```
"你好"
  │
  ▼
┌─────────────────┐
│  Tokenization   │  string → token ids
└─────────────────┘
  │ [108386, 100638]   (2 个 token)
  ▼
┌─────────────────┐
│   Embedding     │  查表: token id → vector
└─────────────────┘
  │ [1, 2, 896]  float16
  ▼
┌─────────────────┐
│ Transformer ×24 │  每层: RMSNorm → Attention → +residual
│                 │        RMSNorm → SwiGLU   → +residual
└─────────────────┘
  │ [1, 2, 896]  形状不变!
  ▼
┌─────────────────┐
│   Final RMSNorm │
└─────────────────┘
  │ [1, 2, 896]
  ▼
┌─────────────────┐
│    LM Head      │  embedding.as_linear: 向量 → 词表 logits
└─────────────────┘
  │ [1, 2, 151936]
  ▼
┌─────────────────┐
│    Sampler      │  top-k / top-p / temperature → 选一个 token
└─────────────────┘
  │ token id (标量)
  ▼
┌─────────────────┐
│ Detokenization  │  token id → string
└─────────────────┘
  │
  ▼
"你"  (第一个新生成的 token)
```

然后进入自回归循环，把新 token 拼回输入，重复上述过程...

---

## 第一步：Tokenization

```
输入: "你好"
  │
  │ tokenizer.encode("你好", add_special_tokens=False)
  ▼
输出: [108386, 100638]        ← 2 个 token id (Qwen2 tokenizer)
       mx.array, shape=(2,)
```

```
"你" → 108386
"好" → 100638
```

这就是模型的"语言"——所有计算都从这两个整数开始。

---

## 第二步：Embedding

```
输入: x = [108386, 100638]    shape (2,)

Embedding.__call__(x):
  return self.weight[x, :]

  weight: (151936, 896)       float16
  x:      (2,)                整数索引

  weight[x, :]  →  高级索引，取出第 108386 行和第 100638 行

输出: h = (2, 896)            float16
       [[ 0.023, -0.154,  0.087, ...,  0.045],    ← "你" 的向量
        [-0.032,  0.201, -0.063, ..., -0.011]]    ← "好" 的向量
```

**然后再加一个 batch 维度**（因为模型期望 `(batch, seq_len, dim)`）：

```python
# 在 simple_generate 中:
h = model(y[None])            # y shape (2,) → y[None] shape (1, 2)
```

```
输入 Embedding:    (2, 896)
加 batch 维:       (1, 2, 896)    ← B=1, L=2, E=896
```

---

## 第三步：Transformer Block × 24

每个 Block 完全相同的结构：

```
输入 x: (1, 2, 896)
  │
  ├─→ RMSNorm ──→ MultiHeadAttention ──→ + ──→ residual = attn_out
  │      ↑              ↑                    ↑
  │   input_layernorm  详细见下             x + attn
  │
  ├─→ RMSNorm ──→ SwiGLU MLP ──→ + ──→ output = mlp_out
  │      ↑              ↑              ↑
  │   post_attn_norm   详细见下       residual + mlp
  │
  ▼
输出: (1, 2, 896)     形状不变
```

### 3.1 RMSNorm（Pre-Attention）

回顾公式：`RMSNorm(x) = x * rsqrt(mean(x²) + eps) * weight`

```
输入 x: (1, 2, 896)   float16

① astype(float32)      → (1, 2, 896)  float32

② square(x)            → (1, 2, 896)   每个元素平方
   mean(..., axis=-1)  → (1, 2, 1)     每个 token 的均方值
   + eps               → (1, 2, 1)     + 1e-5
   rsqrt               → (1, 2, 1)     1/√...

③ x * inv_rms          → (1, 2, 896)   归一化到 RMS≈1

④ * weight             → (1, 2, 896)   可学习的逐维度缩放

⑤ astype(float16)      → (1, 2, 896)  float16
```

```
示意（token "你" 的 896 维向量）:

归一化前:  [0.023, -0.154, 0.087, ..., 0.045]   均方≈1.8
归一化后:  [0.017, -0.114, 0.064, ..., 0.033]   均方≈1.0  (被统一缩放)
缩放后:    [0.018, -0.109, 0.070, ..., 0.031]   weight 微调后
```

**RMSNorm 的作用**：让每个 token 的向量量级一致，避免数值漂移。两个 token 独立归一化。

### 3.2 Multi-Head Attention（GQA）

这是最复杂的一步。输入 `(1, 2, 896)`，输出 `(1, 2, 896)`，但中间经历了大量维度变换。

```
Qwen2-0.5B Attention 参数:
  hidden_size = 896
  num_heads   = 14
  num_kv_heads= 2
  head_dim    = 64
  GQA group   = 14/2 = 7   (每7个Q头共享1对KV头)
```

#### Q/K/V 投影

```
输入 x: (1, 2, 896)

Q 投影:
  linear(x, wq, bq)                     wq: (896, 896)
  → (1, 2, 896)
  reshape → (1, 2, 14, 64)             14 个头，每头 64 维

K 投影:
  linear(x, wk, bk)                     wk: (128, 896)   ← 2×64=128!
  → (1, 2, 128)
  reshape → (1, 2, 2, 64)              只有 2 个 KV 头，每头 64 维

V 投影:
  linear(x, wv, bv)                     wv: (128, 896)
  → (1, 2, 128)
  reshape → (1, 2, 2, 64)
```

**GQA 的关键**：K 和 V 的权重矩阵只有 `num_kv_heads × head_dim = 128` 行，而不是 `num_heads × head_dim = 896` 行。参数更少，KV Cache 更小。

```
Q, K, V 权重矩阵形状对比:

  wq: (14×64, 896) = (896, 896)    14 个 Q 头，每头独立
  wk: ( 2×64, 896) = (128, 896)     2 个 K 头，被 14 个 Q 头共享
  wv: ( 2×64, 896) = (128, 896)     2 个 V 头，被 14 个 Q 头共享
```

#### RoPE 位置编码

```
对 Q 和 K 分别应用 RoPE:

输入 q: (1, 2, 14, 64)
  x1 = q[..., :32]        前半 32 维 = "实部"
  x2 = q[..., 32:]        后半 32 维 = "虚部"

  cos = cos_freqs[:2, :]  (2, 32)    ← 位置 0 和 1 的 cos 值
  sin = sin_freqs[:2, :]  (2, 32)

  real = x1*cos - x2*sin   2D 旋转的实部
  imag = x2*cos + x1*sin   2D 旋转的虚部

  concat → (1, 2, 14, 64)  形状不变

对 k 同样操作: (1, 2, 2, 64) → (1, 2, 2, 64)
```

**为什么只对 Q 和 K 做 RoPE，不对 V？**

RoPE 让 `Q_m · K_n` 的点积值取决于相对位置 `(m-n)`——这正是 Attention 需要的位置信息。V 是"被提取的信息"，不需要位置编码。

#### Q/K/V 重排为多头格式

```
transpose 后:

q: (1, 14, 2, 64)          B=1, H=14, L=2, D=64
k: (1,  2, 2, 64)          B=1, H=2,  L=2, D=64
v: (1,  2, 2, 64)
```

#### Scaled Dot-Product Attention（GQA 版本）

GQA 的核心在 `scaled_dot_product_attention_grouped` 中：

```
q: (1, 14, 2, 64)
k: (1,  2, 2, 64)
v: (1,  2, 2, 64)

H_q=14, H=2, q_counter=14/2=7

① 为 GQA 扩展 KV:
   query = q.reshape(1, 2, 7, 2, 64)     # 2个KV组，每组7个Q头
   key   = k.reshape(1, 2, 1, 2, 64)     # 在每个组内广播
   value = v.reshape(1, 2, 1, 2, 64)

② 计算 Attention Score:
   score = query @ key^T * scale
         = (1, 2, 7, 2, 64) @ (1, 2, 1, 64, 2)
         = (1, 2, 7, 2, 2)              ← (B, H_kv, group, L_q, L_k)

   scale = 1/√64 = 0.125

③ Causal Mask:
   mask = causal_mask(L=2, S=2)
        = [[0,   -∞],
           [0,    0]]

   token 0 只能看到 token 0
   token 1 可以看到 token 0 和 token 1

   score = score + mask

④ Softmax (axis=-1):
   attn_weight = softmax(score)
               = (1, 2, 7, 2, 2)        ← 每行的权重之和 = 1

⑤ Weighted sum:
   attention = attn_weight @ value
             = (1, 2, 7, 2, 2) @ (1, 2, 1, 2, 64)
             = (1, 2, 7, 2, 64)

⑥ reshape 回:
   (1, 2, 7, 2, 64) → (1, 14, 2, 64)
```

**Causal Mask 的直观意义**：

```
"你" (位置0) 查询 "好" (位置1) 时:
  score[0, 1] += -∞  →  softmax(-∞) = 0  →  位置 0 不能 attend 位置 1

"好" (位置1) 查询 "你" (位置0) 时:
  score[1, 0] += 0    →  正常计算        →  位置 1 可以 attend 位置 0
```

在 `generate.py` 中，`mask="causal"` 就是这个意思。

#### Attention 输出

```
① transpose 回:  (1, 14, 2, 64) → (1, 2, 14, 64)
② reshape:       (1, 2, 14, 64) → (1, 2, 896)
③ linear(x, wo): (1, 2, 896)    wo: (896, 896)

输出: (1, 2, 896)
```

#### 残差连接

```python
attn = residual + attn    # (1, 2, 896) + (1, 2, 896) = (1, 2, 896)
```

残差连接确保即使 Attention 输出很差，信息也能绕过去，这是训练深层网络的基础。

### 3.3 RMSNorm（Pre-MLP）

```
输入: (1, 2, 896)  →  RMSNorm →  (1, 2, 896)   均方归一化
```

和 3.1 完全相同的计算，但使用 `post_attention_layernorm` 的独立 weight 参数。

### 3.4 SwiGLU MLP

```
输入: (1, 2, 896)

                                             gate 分支
                                           ┌──────────────┐
                                           │ x @ w_gate^T │   (1,2,896)@(896,4864) → (1,2,4864)
                                           │   silu(·)    │   SiLU 软门控
                                           └──────┬───────┘
                                                  │ (1,2,4864)  值域 [~-0.28, +∞)
   MLP(x):                                        │
                                           ┌──────┼───────┐
                                           │      ⊙       │  ← element-wise 乘法
                                           └──────┬───────┘
                                                  │
                                           ┌──────┴───────┐
                                           │ x @ w_up^T   │   (1,2,896)@(896,4864) → (1,2,4864)
                                           └──────────────┘
                                             up 分支 (无激活)

   gate ⊙ up  →  (1, 2, 4864)

   然后降维:
   (1, 2, 4864) @ w_down^T  →  (1, 2, 896)    w_down: (896, 4864)
```

```
权重形状回顾:
  w_gate: (4864, 896)    门控投影
  w_up:   (4864, 896)    内容投影
  w_down: (896, 4864)    降维投影
```

#### 残差连接

```python
mlp = residual + mlp       # (1, 2, 896) + (1, 2, 896) = (1, 2, 896)
```

---

## Block 内完整维度流转

```
Block 输入:  x  (1, 2, 896)
  │
  ├─ residual = x
  │
  ├─ RMSNorm(input_layernorm)
  │   (1,2,896) → (1,2,896)
  │
  ├─ Attention
  │   Q: (1,2,896) → (1,14,2,64)    14 头 Q
  │   K: (1,2,896) → (1, 2,2,64)     2 头 K  (GQA!)
  │   V: (1,2,896) → (1, 2,2,64)     2 头 V
  │   RoPE(Q,K)                       注入位置
  │   Score: Q·K^T·scale → (1,2,7,2,2)  causal mask + softmax
  │   Output: attn@V → (1,14,2,64)
  │   wo 投影 → (1,2,896)
  │
  ├─ x = residual + attn_out          ← 第一次残差
  │
  ├─ residual = x
  │
  ├─ RMSNorm(post_attention_layernorm)
  │   (1,2,896) → (1,2,896)
  │
  ├─ SwiGLU MLP
  │   gate: (1,2,896) → (1,2,4864) → silu
  │   up:   (1,2,896) → (1,2,4864)
  │   gate⊙up → (1,2,4864)
  │   down:  (1,2,4864) → (1,2,896)
  │
  ├─ x = residual + mlp_out           ← 第二次残差
  │
  ▼
Block 输出:  (1, 2, 896)              ← 形状完全不变!
```

**关键观察**：Transformer Block 像一个"形状保持过滤器"——输入什么形状，输出什么形状。这是残差连接的基础：`x + f(x)` 要求 `f(x)` 和 `x` 同形状。

---

## 第四步：Final RMSNorm

24 层之后，最后做一次 RMSNorm：

```
输入:  (1, 2, 896)   经过 24 层变换后的 hidden states
输出:  (1, 2, 896)   归一化到 RMS≈1

这是 model.norm，用于稳定最后的 lm_head 投影。
```

---

## 第五步：LM Head

因为 `tie_word_embeddings=True`，使用 `embedding.as_linear`：

```python
# Qwen2ModelWeek1.__call__:
if self.w_lm_head is not None:
    return linear(h, self.w_lm_head)
else:
    return self.embedding.as_linear(h)    # ← tie_word_embeddings=True 走这里
```

```
as_linear(h) = linear(h, self.weight)
             = h @ weight^T

h:      (1, 2, 896)
weight: (151936, 896)       ← 就是 embedding 的同一份权重!
weight^T: (896, 151936)

h @ weight^T  →  (1, 2, 896) @ (896, 151936) = (1, 2, 151936)
```

**输出含义**：`(1, 2, 151936)` — 2 个 token 位置，每个位置有 151936 个 logits（每个词表 token 的得分）。

```
位置 0 ("你"):  [ -2.3,   5.1,  -0.8, ...,   8.7, ... ]  151936 个数
位置 1 ("好"):  [  1.2,  -3.4,   6.2, ...,  -0.3, ... ]  151936 个数
                                                      ↑
                                              第 108386 个 = "你" 的得分
```

---

## 第六步：Sampler

在 `simple_generate` 中，我们只关心**最后一个位置**的 logits（因为是预测下一个 token）：

```python
def _step(model, y):
    logits = model(y[None])           # (1, L, 151936)
    logits = logits[:, -1, :]         # (1, 151936)  ← 只取最后一个位置!
    logprobs = logits - mx.logsumexp(logits, keepdims=True)   # log-softmax
```

**log-softmax 的数学**：

```
logprobs_i = logits_i - log(Σ_j exp(logits_j))

这等价于:  logprobs = log(softmax(logits))
```

```
示意（取前 5 个维度）:

logits:   [-2.3,   5.1,  -0.8,   2.3,   8.7, ...]
softmax:  [1e-5,  0.026, 2e-4,  0.002, 0.845, ...]
logprobs: [-11.5, -3.65, -8.51, -6.21, -0.17, ...]
```

### 采样策略

```python
def make_sampler(temp=0.7, top_p=0.9, top_k=50):
    def sample(logprobs):
        # ① top-k 过滤
        if top_k > 0:
            # 保留 logprobs 最大的 50 个 token
            # 其余设为 -inf (概率=0)

        # ② top-p (nucleus) 过滤
        if top_p > 0:
            # 按概率从大到小排序
            # 累积概率超过 0.9 以后的 token 全部设 -inf

        # ③ temperature 缩放
        logprobs = logprobs / temp   # temp<1 让分布更尖锐 (更确定)
                                      # temp>1 让分布更平坦 (更多样)

        # ④ 采样
        return mx.random.categorical(logprobs)
```

```
采样过程示意:

logprobs 最大的前 5 个 token:
  token 108386 ("你"):  -0.17  →  exp(-0.17) ≈ 0.844
  token  45678 ("。"):   -2.30  →  exp(-2.30) ≈ 0.100
  token  23456 ("!"):    -3.00  →  exp(-3.00) ≈ 0.050
  token  87654 (","):    -4.60  →  exp(-4.60) ≈ 0.010
  token  34567 ("的"):   -5.80  →  exp(-5.80) ≈ 0.003

  其余 151931 个 token:  概率极小或被 top-k/top-p 截断

  按概率随机抽取 → 大概率抽到 108386 ("你")
```

- **temp=0**：退化为 argmax（贪心解码），永远选概率最大的
- **temp=0.7**：温和随机，输出较连贯
- **temp=2.0**：高度随机，输出多样但可能不连贯

---

## 第七步：Detokenization

```python
token = sampler(logprobs)         # 例如抽到 108386

detokenizer.add_token(token.item())
print(detokenizer.last_segment, end="", flush=True)
# 输出: "你"
```

tokenizer 的 detokenizer 维护内部状态，增量解码新 token。

---

## 第八步：自回归循环

第一个新 token 生成后，拼回序列，继续：

```python
# simple_generate 的主循环
tokens = mx.array(tokenizer.encode("你好"))      # (2,)

while True:
    token = _step(model, tokens)                  # 生成 1 个新 token
    tokens = mx.concat([tokens, token])           # 拼回 (3,) → (4,) → ...
    if token.item() == tokenizer.eos_token_id:    # 遇到结束符退出
        break
```

```
完整生成过程:

输入:  "你好"        →  tokens: [108386, 100638]            L=2
第1步: model(2 tokens)→  logits[最后位置] → 采样 → 108386    "你"
       concat         →  tokens: [108386, 100638, 108386]   L=3
第2步: model(3 tokens)→  logits[最后位置] → 采样 → 100009    "好"
       concat         →  tokens: [108386, 100638, 108386, 100009]  L=4
第3步: model(4 tokens)→  logits[最后位置] → 采样 → EOS      结束
```

**注意**：每一步输入变长，Transformer 重新计算所有位置（没有 KV Cache 时，Week 1 实现），但只取最后一个位置的 logits 用于采样。

---

## 完整数据流总图

```
                          generate.py                          model internals
 ┌──────────────────────────────────────┐    ┌──────────────────────────────────────────────┐
 │                                      │    │                                              │
 │  prompt = "你好"                      │    │                                              │
 │                                      │    │                                              │
 │  tokens = tokenizer.encode(prompt)   │    │                                              │
 │         = [108386, 100638]   (2,)    │    │                                              │
 │                                      │    │                                              │
 │  ┌─────────────────────────────┐     │    │                                              │
 │  │       while True:           │     │    │                                              │
 │  │                             │     │    │                                              │
 │  │  token = _step(model,tokens)│─────┼───→│  model(tokens[None])                        │
 │  │                             │     │    │    tokens: (1, 2)                            │
 │  │                             │     │    │                                              │
 │  │                             │     │    │  embedding(tokens)                          │
 │  │                             │     │    │    weight[tokens] → (1, 2, 896)             │
 │  │                             │     │    │                                              │
 │  │                             │     │    │  for layer in 0..23:                        │
 │  │                             │     │    │    ┌─────────────────────────┐               │
 │  │                             │     │    │    │ RMSNorm → (1,2,896)     │               │
 │  │                             │     │    │    │                         │               │
 │  │                             │     │    │    │ Q:(1,2,896)→(1,14,2,64)│               │
 │  │                             │     │    │    │ K:(1,2,896)→(1, 2,2,64)│ GQA           │
 │  │                             │     │    │    │ V:(1,2,896)→(1, 2,2,64)│               │
 │  │                             │     │    │    │                         │               │
 │  │                             │     │    │    │ RoPE(Q), RoPE(K)        │ 位置编码      │
 │  │                             │     │    │    │                         │               │
 │  │                             │     │    │    │ Q·K^T·scale             │               │
 │  │                             │     │    │    │   + causal mask         │               │
 │  │                             │     │    │    │   + softmax             │               │
 │  │                             │     │    │    │   @ V                   │               │
 │  │                             │     │    │    │ → (1,14,2,64)          │               │
 │  │                             │     │    │    │ wo:(1,14,2,64)         │               │
 │  │                             │     │    │    │     → (1,2,896)        │               │
 │  │                             │     │    │    │ + residual             │ 残差          │
 │  │                             │     │    │    │                         │               │
 │  │                             │     │    │    │ RMSNorm → (1,2,896)    │               │
 │  │                             │     │    │    │ SwiGLU  → (1,2,896)    │               │
 │  │                             │     │    │    │ + residual             │ 残差          │
 │  │                             │     │    │    └─────────────────────────┘               │
 │  │                             │     │    │                                              │
 │  │                             │     │    │  final RMSNorm → (1, 2, 896)                │
 │  │                             │     │    │                                              │
 │  │                             │     │    │  lm_head: h @ weight^T                      │
 │  │                             │     │    │    (1,2,896)@(896,151936) → (1,2,151936)   │
 │  │                             │     │    │                                              │
 │  │  ←──── logits ──────────────┼─────┼────│                                              │
 │  │                             │     │    │                                              │
 │  │  logits = logits[:,-1,:]    │     │    │                                              │
 │  │         = (1, 151936)       │     │    │                                              │
 │  │                             │     │    │                                              │
 │  │  logprobs = logits          │     │    │                                              │
 │  │    - logsumexp(logits)      │     │    │                                              │
 │  │                             │     │    │                                              │
 │  │  token = sampler(logprobs)  │     │    │                                              │
 │  │        → 标量 (例如 108386)  │     │    │                                              │
 │  │                             │     │    │                                              │
 │  │  tokens = concat(           │     │    │                                              │
 │  │    [tokens, token])         │     │    │                                              │
 │  │  → (3,) → (4,) → ...       │     │    │                                              │
 │  │                             │     │    │                                              │
 │  │  detokenizer.add_token()    │     │    │                                              │
 │  │  print("你")                 │     │    │                                              │
 │  │                             │     │    │                                              │
 │  │  if token == EOS: break     │     │    │                                              │
 │  │                             │     │    │                                              │
 │  └─────────────────────────────┘     │    │                                              │
 │                                      │    │                                              │
 │  输出: "你好啊！今天天气..."            │    │                                              │
 └──────────────────────────────────────┘    └──────────────────────────────────────────────┘
```

---

## 各阶段维度速查表

| 阶段 | 输入形状 | 输出形状 | 说明 |
|------|---------|---------|------|
| Tokenization | `"你好"` | `(2,)` | string → token ids |
| Embedding | `(2,)` | `(2, 896)` | 查表 |
| add batch dim | `(2, 896)` | `(1, 2, 896)` | model 需要 batch 维 |
| RMSNorm | `(1, 2, 896)` | `(1, 2, 896)` | 逐 token 归一化 |
| Q 投影 | `(1, 2, 896)` | `(1, 14, 2, 64)` | 14 头 |
| K 投影 | `(1, 2, 896)` | `(1, 2, 2, 64)` | 2 头 (GQA) |
| V 投影 | `(1, 2, 896)` | `(1, 2, 2, 64)` | 2 头 (GQA) |
| RoPE | `(1, 14, 2, 64)` | `(1, 14, 2, 64)` | 形状不变 |
| Attention Score | `Q·K^T` | `(1, 2, 7, 2, 2)` | GQA 分组 |
| Attention Output | - | `(1, 14, 2, 64)` | 加权求和 |
| wo 投影 | `(1, 14, 2, 64)` | `(1, 2, 896)` | reshape + linear |
| + residual | `(1, 2, 896)` | `(1, 2, 896)` | 逐元素加 |
| RMSNorm | `(1, 2, 896)` | `(1, 2, 896)` | |
| SwiGLU gate | `(1, 2, 896)` | `(1, 2, 4864)` | 膨胀 |
| SwiGLU up | `(1, 2, 896)` | `(1, 2, 4864)` | 膨胀 |
| gate ⊙ up | `(1, 2, 4864)` | `(1, 2, 4864)` | element-wise |
| w_down 投影 | `(1, 2, 4864)` | `(1, 2, 896)` | 压缩 |
| + residual | `(1, 2, 896)` | `(1, 2, 896)` | |
| ×24 层后 | `(1, 2, 896)` | `(1, 2, 896)` | 形状始终不变 |
| Final RMSNorm | `(1, 2, 896)` | `(1, 2, 896)` | |
| LM Head | `(1, 2, 896)` | `(1, 2, 151936)` | 投影到词表 |
| 取最后位置 | `(1, 2, 151936)` | `(1, 151936)` | 只要最后一个 |
| log-softmax | `(1, 151936)` | `(1, 151936)` | 转为 log 概率 |
| Sampler | `(1, 151936)` | 标量 | top-k/p + temp + categorical |
| Detokenize | 标量 | `"你"` | token → string |
| Concat | `(L,)` + 标量 | `(L+1,)` | 拼回序列 |

---

## 核心洞察

1. **Embedding 层是模型的"字典"** — 把离散 token 映射到连续向量空间，并复用做 lm_head（weight tying）

2. **Transformer Block 是形状保持器** — 输入 `(B,L,D)`，输出 `(B,L,D)`。内部膨胀到 `intermediate_size`（MLP）和多头（Attention），但最终都压缩回去

3. **残差连接是高速公路** — `x = x + f(x)`，即使 `f(x)` 输出接近 0，x 本身也能无损传递

4. **GQA 是性价比设计** — 14 个 Q 头共享 2 对 KV 头，KV Cache 减少 7×，推理速度提升

5. **RoPE 只编码 Q 和 K** — 让 Attention Score 携带相对位置信息，V 不需要旋转

6. **自回归是串行瓶颈** — 每生成一个新 token 都要重跑整个模型，这是 LLM 推理慢的根本原因（KV Cache 可以缓解）

7. **用 logprobs 而非 probs** — 数值更稳定：`logsumexp` 避免 exp 溢出，top-k 在 log 空间做过滤
