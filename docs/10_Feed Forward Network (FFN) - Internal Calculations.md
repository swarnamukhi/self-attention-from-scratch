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

---

# Step 1: Input to the Feed Forward Network

Suppose the sentence is

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

This vector is the input to the Feed Forward Network.

Its shape is

```text
1 × 4
```

In the original Transformer paper, this would be

```text
1 × 512
```

For simplicity, we use an embedding dimension of **4** instead of **512** while preserving the same architecture.

---

# Step 2: First Linear Layer

The first operation inside the FFN is a **Linear Layer**.

According to the original Transformer paper,

```text
Input

↓

Linear

↓

Activation

↓

Linear

↓

Output
```

Since our input embedding contains **4 features**, and we want to expand it to **16 hidden features**, the first weight matrix must have the following shape.

```text
Input Shape

1 × 4

×

Weight Matrix (W₁)

4 × 16

=

Output

1 × 16
```

Notice that the **number of columns in the input (4)** must match the **number of rows in the weight matrix (4)**.

This is the basic rule of matrix multiplication.

Therefore,

```text
(1 × 4)

×

(4 × 16)

=

(1 × 16)
```

The FFN has now expanded the embedding from **4 dimensions** to **16 dimensions**.

This is exactly the same operation performed in the original Transformer.

```text
Paper

(1 × 512)

×

(512 × 2048)

=

(1 × 2048)
```

Only the dimensions have been reduced for easier manual calculation.

---

# Step 3: Constructing the First Weight Matrix (W₁)

To understand the calculations, let us create a simple weight matrix.

```

W₁

| Input Feature | H₁ | H₂ | H₃ | H₄ | H₅ | H₆ | H₇ | H₈ | H₉ | H₁₀ | H₁₁ | H₁₂ | H₁₃ | H₁₄ | H₁₅ | H₁₆ |
|--------------|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|-----|
| x₁ | 0.2 | 0.1 | 0.3 | 0.4 | 0.5 | 0.2 | 0.1 | 0.3 | 0.4 | 0.2 | 0.1 | 0.3 | 0.5 | 0.2 | 0.4 | 0.1 |
| x₂ | 0.1 | 0.3 | 0.2 | 0.5 | 0.2 | 0.4 | 0.3 | 0.1 | 0.5 | 0.3 | 0.2 | 0.4 | 0.1 | 0.5 | 0.2 | 0.3 |
| x₃ | 0.4 | 0.2 | 0.5 | 0.1 | 0.3 | 0.5 | 0.2 | 0.4 | 0.1 | 0.5 | 0.3 | 0.2 | 0.4 | 0.1 | 0.5 | 0.2 |
| x₄ | 0.3 | 0.4 | 0.1 | 0.2 | 0.4 | 0.1 | 0.5 | 0.2 | 0.3 | 0.1 | 0.4 | 0.5 | 0.2 | 0.3 | 0.1 | 0.4 |

```

Here,

- Rows represent the **input features**.
- Columns represent the **16 hidden neurons**.

Every hidden neuron has one weight connected to each input feature.

For example,

Hidden Neuron **H₁** uses the weights

```text
[0.2, 0.1, 0.4, 0.3]
```

Similarly,

Hidden Neuron **H₂** uses

```text
[0.1, 0.3, 0.2, 0.4]
```

We will now compute the output of each hidden neuron one by one.

---

# Step 4: Calculating the Output of One Hidden Neuron

The output of every hidden neuron is calculated using the same process.

Instead of calculating all **16 neurons**, we will calculate **only the first hidden neuron (H₁)**.

The remaining neurons follow exactly the same procedure using their respective weights.

Suppose,

Input embedding

```text
X = [2, 1, 3, 4]
```

Weights connected to **H₁**

```text
W₁(H₁) = [0.2, 0.1, 0.4, 0.3]
```

Bias

```text
b₁ = 0.5
```

The output of the first hidden neuron is

```text
H₁

= (2 × 0.2)
+ (1 × 0.1)
+ (3 × 0.4)
+ (4 × 0.3)
+ 0.5
```

```text
= 0.4
+ 0.1
+ 1.2
+ 1.2
+ 0.5
```

```text
= 3.4
```

Therefore,

```text
H₁ = 3.4
```

The remaining hidden neurons (**H₂ to H₁₆**) are calculated in exactly the same way using their corresponding weight values.

After computing all 16 neurons, the output of the first Linear layer becomes

```text
[H₁, H₂, H₃, ..., H₁₆]
```

whose shape is

```text
1 × 16
```
---

# Step 5: Output of the First Linear Layer

Using the same calculation process, we compute the outputs of the remaining hidden neurons (**H₂ to H₁₆**).

For simplicity, let us assume the outputs are as follows:

```text
H₁  = 3.4
H₂  = -2.8
H₃  = 4.1
H₄  = -1.2
H₅  = 4.5
H₆  = 2.6
H₇  = -0.7
H₈  = 2.9
H₉  = 4.2
H₁₀ = -3.1
H₁₁ = 2.7
H₁₂ = 3.9
H₁₃ = 4.4
H₁₄ = -2.5
H₁₅ = 3.6
H₁₆ = 1.8
```

The output of the first Linear layer is therefore

```text
[3.4, -2.8, 4.1, -1.0, 4.5, 2.6, -0.7, 2.9,
 4.2, -3.1, 2.7, 3.9, 4.4, -2.5, 3.6, 1.8]
```

The shape of this output is

```text
1 × 16
```

This expanded representation now becomes the input to the activation function.

---

# Step 6: Activation Functions

The output of the first Linear layer is passed to an **activation function**.

Suppose the output of the first Linear layer is

```text
[3.4, -2.8, 4.1, -1.2, 4.5, 2.6, -0.7, 2.9,
 4.2, -3.1, 2.7, 3.9, 4.4, -2.5, 3.6, 1.8]
```

The activation function is applied **independently to each hidden neuron**.

The output shape does not change.

```text
Before Activation : 1 × 16

↓

Activation Function

↓

After Activation : 1 × 16
```

Different Transformer models use different activation functions.

---

## 1. ReLU (Rectified Linear Unit)

The original **Attention Is All You Need** paper uses **ReLU**.

The ReLU function is

```text
ReLU(x) = max(0, x)
```

This means

- Positive values remain unchanged.
- Negative values become **0**.

Example

```text
Before ReLU

[3.4, -2.8, 4.1, -1.2]
```

```text
After ReLU

[3.4, 0, 4.1, 0]
```

ReLU completely removes negative values.

---

## 2. GELU (Gaussian Error Linear Unit)

Most modern language models use **GELU** instead of ReLU.

Unlike ReLU, GELU does not immediately discard negative values.

Instead, it smoothly scales them.

Example (approximately)

```text
Before GELU

[3.4, -2.8, 4.1, -1.2]
```

```text
After GELU

[3.39, -0.01, 4.10, -0.14]
```

Notice that

- Large positive values remain almost unchanged.
- Small negative values are reduced but not completely removed.
- Information is preserved more smoothly.

This often leads to better learning in deep Transformer models.

---

## 3. SwiGLU

Many recent Large Language Models (LLMs) such as **LLaMA** use **SwiGLU**.

Unlike ReLU and GELU, SwiGLU does not simply apply an activation to the input.

Instead, it creates **two parallel projections** from the first Linear layer.

```text
Hidden Representation

        │
   ┌────┴────┐
   ▼         ▼
Linear A   Linear B
   │         │
   ▼         ▼
SiLU      Identity
   │         │
   └────┬────┘
        ▼
 Element-wise Multiplication
        │
        ▼
      Output
```

This gating mechanism allows the model to decide **which information should pass through and which should be suppressed**, making the FFN more expressive.

---

## Comparison

| Activation | How it works | Used in |
|------------|--------------|---------|
| ReLU | Removes all negative values | Original Transformer (2017) |
| GELU | Smoothly scales values instead of hard clipping | BERT, GPT-2, GPT-3, Gemma |
| SwiGLU | Uses a gating mechanism with two projections | LLaMA, PaLM, Mistral |

---

## Key Takeaway

All activation functions are applied after the first Linear layer.

```text
Linear

↓

Activation

↓

Linear
```

The FFN architecture remains the same.

Only the activation function changes depending on the Transformer model.
