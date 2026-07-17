# 10. Feed Forward Network (FFN) - Internal Calculations

## Introduction

In the previous chapter, we learned the theory behind the Feed Forward Network (FFN) and understood why every Transformer encoder contains an FFN after the Multi-Head Self-Attention block.

Now, we will go one step further.

In this chapter, we will perform the **internal calculations of the Feed Forward Network** exactly as we did for Self-Attention.

Our objective is to understand:

- How the FFN works internally.
- How matrix multiplication is performed.
- Why two Linear layers are used.
- Why an activation function is required.
- How dimensions change inside the FFN.
- How the implementation relates to the original **"Attention Is All You Need"** paper.

By the end of this chapter, you will be able to understand the FFN equation used in research papers as well as its implementation in TensorFlow and PyTorch.

---

# Where are we inside the Transformer?

The Feed Forward Network is the second major sub-layer inside every Transformer Encoder.

```text
Input Embedding + Positional Encoding
                │
                ▼
      Multi-Head Self-Attention
                │
                ▼
          Add + LayerNorm
                │
                ▼
      Feed Forward Network
                │
                ▼
          Add + LayerNorm
                │
                ▼
      Output of Encoder Layer
```

By the time the data reaches the FFN,

- Self-Attention has already been completed.
- Every token already knows the context of the sentence.
- The first Residual Connection and Layer Normalization have already been applied.

Therefore, the FFN does **not** perform communication between tokens.

Instead, it improves the representation of **each token independently**.

---

# FFN Equation from the Original Paper

The original Transformer paper defines the Feed Forward Network as

\[
FFN(x)=\max(0,xW_1+b_1)W_2+b_2
\]

At first glance, this equation looks complicated.

However, it is actually performing only three operations.

```text
Input
   │
   ▼
Linear Layer
(xW₁ + b₁)
   │
   ▼
ReLU
max(0,x)
   │
   ▼
Linear Layer
(...W₂ + b₂)
   │
   ▼
Output
```

This is exactly the same computation implemented in TensorFlow.

```python
Dense(2048, activation="relu")
Dense(512)
```

The only difference is the notation.

Research papers describe the operations mathematically.

TensorFlow describes the same operations using programming syntax.

---

# Dimensions Used in the Original Transformer Paper

The original paper defines two important dimensions.

| Symbol | Meaning | Value |
|---------|---------|------:|
| d_model | Embedding Dimension | 512 |
| d_ff | Hidden Dimension | 2048 |

Therefore, one token flows through the FFN as

```text
1 × 512

↓

Linear

↓

1 × 2048

↓

ReLU

↓

1 × 2048

↓

Linear

↓

1 × 512
```

Notice that the embedding temporarily expands from **512 dimensions** to **2048 dimensions** before returning to **512 dimensions**.

---

# Why Don't We Calculate Using 512 and 2048?

The first Linear layer requires a weight matrix of

```text
512 × 2048
```

which contains

```text
512 × 2048

=

1,048,576 weights
```

Calculating more than one million weights manually is impossible.

Therefore, throughout this chapter, we will use a smaller example while preserving exactly the same architecture.

---

# Scaled-Down Example

Instead of

```text
512

↓

2048

↓

512
```

we will use

```text
4

↓

16

↓

4
```

Notice that

```text
2048 = 4 × 512
```

Similarly,

```text
16 = 4 × 4
```

The expansion ratio remains exactly the same.

Therefore,

although the numbers are smaller,

the mathematics is identical to the original Transformer.

---

# Example Sentence

Suppose our sentence is

```text
I love cats
```

After

- Multi-Head Self-Attention
- First Residual Connection
- First Layer Normalization

the contextual embedding of the token **love** becomes

```text
love

↓

[2, 1, 3, 4]
```

This vector will become the input to the Feed Forward Network.

---

# One Important Observation

Many beginners think the FFN receives the entire sentence.

This is incorrect.

The FFN processes **one token at a time**.

For example,

```text
I

↓

FFN
```

```text
love

↓

FFN
```

```text
cats

↓

FFN
```

Each token passes independently through the same Feed Forward Network.

The weights remain identical.

Only the input embedding changes.

---

# Input Dimension

Our input embedding is

```text
[2, 1, 3, 4]
```

Therefore,

its shape is

```text
1 × 4
```

In the original paper, this would have been

```text
1 × 512
```

---

# Next Step

In the next section, we will begin the actual calculations.

We will

- construct the first weight matrix (W₁),
- define the bias vector (b₁),
- perform matrix multiplication,
- calculate every hidden neuron,
- apply the ReLU activation function,
- understand how the embedding expands from **4 dimensions** to **16 dimensions**.

This is exactly the same process performed by the original Transformer architecture, only with smaller dimensions for easier manual calculation.
