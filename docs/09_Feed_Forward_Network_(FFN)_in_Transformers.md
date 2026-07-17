# Feed Forward Network (FFN) in Transformers

## Introduction

After Multi-Head Self-Attention, every token already understands the context of the entire sentence.

For example,

Sentence:

I love cats

After Self-Attention,

the embedding of **love** already contains information about:

- I
- love
- cats

This new embedding is called a **Context-Aware Embedding**.

At this stage, the model does not need any more communication between tokens.

Instead, it needs to **improve the representation of each token individually.**

This is the purpose of the **Feed Forward Network (FFN).**

---

# Where does FFN appear?

One Transformer Encoder Layer consists of:

```
Input Embedding
        │
        ▼
Multi-Head Self-Attention
        │
        ▼
Residual Add
        │
        ▼
Layer Normalization
        │
        ▼
Feed Forward Network (FFN)
        │
        ▼
Residual Add
        │
        ▼
Layer Normalization
        │
        ▼
Output Encoder Layer
```

---

# Why is FFN Required?

Self-Attention performs one job:

> Communication

Every token communicates with every other token.

Example

```
love

↓

Learns information from

I

cats
```

After communication,

the token has gathered information.

However,

it has **not yet transformed or refined** this information.

The FFN performs this second step.

---

# Communication vs Computation

Self-Attention answers

> Which words should I pay attention to?

FFN answers

> Now that I know everything, how should I improve my own representation?

Therefore,

Self-Attention = Communication

FFN = Computation

---

# Real-Life Analogy

Imagine four students working on a project.

Step 1

Every student shares knowledge.

Student A learns from B.

Student A learns from C.

Student A learns from D.

This is similar to **Self-Attention**.

Step 2

Now Student A sits alone and analyzes all the collected information.

This is similar to the **Feed Forward Network**.

---

# Is FFN Different from ANN?

No.

This is one of the biggest misconceptions.

A Feed Forward Network **is simply a small Artificial Neural Network (ANN).**

Researchers use the name "Feed Forward Network" because information only moves forward.

There is

- no recurrence
- no attention
- no feedback loop

---

# ANN vs FFN

Traditional ANN

```
Age
Salary
Credit Score
        │
        ▼
Dense Layer
        │
        ▼
ReLU
        │
        ▼
Dense Layer
        │
        ▼
Prediction
```

Transformer FFN

```
One Token Embedding
        │
        ▼
Linear Layer
        │
        ▼
Activation
        │
        ▼
Linear Layer
        │
        ▼
Refined Token Embedding
```

Notice that the architectures are almost identical.

The difference is **where they are used.**

---

# Important Difference

ANN receives

```
Age
Salary
Credit Score
```

all together.

The FFN receives only

```
One Token Embedding
```

For example

```
love

↓

[0.2 0.8 0.4 0.7]
```

Only this token passes through the FFN.

No interaction with other tokens happens inside the FFN.

---

# Why?

Because communication has already happened inside Multi-Head Attention.

The FFN only improves the current token representation.

---

# Structure of FFN

Mathematically,

```
Linear

↓

Activation

↓

Linear
```

In TensorFlow/Keras

```python
Dense(2048, activation="relu")

Dense(512)
```

or

```python
Dense(2048, activation="gelu")

Dense(512)
```

These are exactly equivalent.

---

# What is a Linear Layer?

A Linear Layer performs

Y = XW + b

where

X = Input

W = Weight Matrix

b = Bias

In TensorFlow,

```
Dense(...)
```

is simply another name for a Linear Layer.

---

# Why is Activation Required?

Suppose we write

```
Linear

↓

Linear
```

Two consecutive linear transformations are mathematically equivalent to a single linear transformation.

Therefore,

without an activation function,

adding multiple Linear layers provides little additional expressive power.

The activation function introduces non-linearity.

Common choices:

- ReLU
- GELU
- SwiGLU

---

# Why Expand the Dimension?

Suppose

Embedding Dimension = 512

The FFN becomes

```
512

↓

2048

↓

512
```

Why?

The first Linear layer expands the feature space.

Think of it as giving the network more room to learn richer feature combinations.

---

# Why Compress Again?

The next Encoder Layer expects

Embedding Dimension = 512.

Therefore,

the FFN returns the embedding back to 512 dimensions.

Also,

Residual Connections require both vectors to have identical dimensions.

---

# Why 2048?

There is no mathematical rule.

The original Transformer paper used

```
Embedding Size = 512

Hidden Size = 2048
```

Notice

```
2048 = 4 × 512
```

Modern Transformers often use

```
Hidden Size ≈ 4 × Embedding Size
```

This provides a good balance between

- learning capacity
- computation
- memory

---

# Can We Use 512 → 512?

Yes.

```
512

↓

512

↓

512
```

The model still works.

However,

it has

- fewer parameters
- lower learning capacity

Therefore,

performance is usually lower.

---

# Why Not 10000?

A very large hidden dimension increases

- memory usage
- computation time
- overfitting risk

Researchers found that approximately

```
Hidden Size ≈ 4 × Embedding Size
```

works well.

---

# Does Every Token Have a Different FFN?

No.

Suppose

```
I

love

cats
```

Each token passes through the FFN independently.

```
I
 │
 ▼
FFN
 │
 ▼

love
 │
 ▼
FFN
 │
 ▼

cats
 │
 ▼
FFN
```

All three use **the same weights**.

Only the inputs are different.

---

# Does the FFN Mix Tokens?

No.

Self-Attention mixes information across tokens.

The FFN processes one token at a time.

---

# Typical Flow

```
Token Embedding
        │
        ▼
Linear (512 → 2048)
        │
        ▼
ReLU / GELU
        │
        ▼
Linear (2048 → 512)
        │
        ▼
Residual Add
        │
        ▼
Layer Normalization
        │
        ▼
Next Encoder Layer
```

---

# Key Takeaways

✅ Self-Attention performs communication between tokens.

✅ FFN performs computation on each token independently.

✅ FFN is simply a small ANN.

✅ Dense Layer = Linear Layer.

✅ Activation introduces non-linearity.

✅ The hidden dimension is usually expanded by approximately 4×.

✅ The output dimension returns to the embedding size for the next Encoder layer.

✅ The same FFN is applied independently to every token.

---

# One-Line Summary

**Self-Attention lets every token gather information from other tokens, while the Feed Forward Network refines each token's representation independently before passing it to the next encoder layer.**
