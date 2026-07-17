# 08_Normalization_in_Deep_Learning_and_Transformers.md

# Why Do We Need Normalization?

Normalization is one of the most confusing topics in Machine Learning and Deep Learning because the same word **"Normalization"** is used in different contexts.

There are **three different concepts** that students often confuse:

1. Data Normalization / Standardization
2. Batch Normalization
3. Layer Normalization

Although all three aim to improve model training, they are applied at different stages and solve different problems.

---

# 1. Data Normalization / Standardization (Before Training)

Suppose we have the following dataset.

| Age | Salary | Experience |
|------|---------|------------|
|25|30,000|2|
|35|80,000|8|
|45|150,000|20|

Notice the feature scales.

```
Age

25 - 45
```

```
Salary

30,000 - 150,000
```

```
Experience

2 - 20
```

The feature **Salary** has much larger values than the others.

During gradient descent,

large-scale features may dominate the learning process.

Therefore, before training we preprocess the data.

---

## Standardization

The most common preprocessing method is Standardization.

Formula

```
z = (x - μ) / σ
```

where

- μ = Mean
- σ = Standard Deviation

After Standardization

- Mean becomes approximately 0
- Standard Deviation becomes approximately 1

Example

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

---

## Min-Max Normalization

Another preprocessing technique is

```
x' = (x - min)/(max - min)
```

which scales every feature between

```
0

and

1
```

---

# When Do We Use Data Normalization?

Whenever the input consists of numerical features having different scales.

Examples

- Age
- Salary
- Height
- Weight
- Temperature
- Sensor Data

This preprocessing is performed **once before training**.

---

# Do We Normalize Text?

No.

Suppose we have

```
I love cats
```

Words do not have numerical scales like

```
Age = 30

Salary = 100000
```

Instead,

the text is converted into

```
Tokens

↓

Embeddings
```

Therefore,

StandardScaler or MinMaxScaler are **not applied to words**.

---

# 2. Batch Normalization

After the input enters a neural network,

every hidden layer produces activations.

Example

Layer Output

```
[250, -180, 410, 95]
```

These values become the input to the next layer.

As the network becomes deeper,

the activations may become

```
very large
```

or

```
very small
```

making training unstable.

Batch Normalization solves this problem.

---

## How Batch Normalization Works

Suppose

Batch Size = 4

Hidden Layer Output

| Sample | Feature1 | Feature2 | Feature3 |
|---------|----------|----------|----------|
|S1|2|5|8|
|S2|4|6|7|
|S3|3|8|9|
|S4|5|7|10|

BatchNorm computes

Mean and Standard Deviation

using **all samples in the mini-batch**.

For Feature1

```
2
4
3
5
```

Mean

↓

Normalize

The same process is repeated for every feature.

---

## Characteristics of BatchNorm

- Applied during training
- Uses mini-batch statistics
- Common in ANN and CNN
- Improves training stability

Typical CNN block

```
Conv

↓

BatchNorm

↓

ReLU

↓

Pooling
```

---
# What Does "Activation" Mean?

One of the biggest confusions in Deep Learning is the word **activation**.

Many beginners think

```
Activation

=

Output after ReLU
```

This is **not completely correct**.

In Deep Learning,

an **activation** simply means

> **The output produced by any neural network layer.**

The layer may or may not contain an activation function.

---

## Example 1 : ANN

```
Input

↓

Dense Layer

↓

ReLU

↓

Output
```

The output after ReLU is called an activation.

---

## Example 2 : Transformer

The Transformer Encoder begins with

```
Embedding

↓

Positional Encoding

↓

Multi-Head Attention
```

Notice

There is **no ReLU** here.

Suppose the embedding of the word

```
love
```

is

```
[5,1,2,3]
```

After Multi-Head Attention,

suppose we obtain

```
[2,4,3,5]
```

This vector is already called an **activation** because it is the output of the Multi-Head Attention layer.

No activation function has been applied.

---

## Residual Connection

Transformer now adds the original input.

```
Input

[5,1,2,3]

+

Attention Output

[2,4,3,5]

=

[7,5,5,8]
```

This new vector

```
[7,5,5,8]
```

is also an activation because it is the output of the Residual Add operation.

---

## Layer Normalization

LayerNorm receives

```
[7,5,5,8]
```

It computes

- Mean
- Standard Deviation

and produces

```
[0.58,-0.96,-0.96,1.35]
```

LayerNorm is **normalizing the activation**, i.e., the output produced by the previous layer.

---

## Feed Forward Network

After LayerNorm,

the vector enters the Feed Forward Network.

```
LayerNorm

↓

Dense

↓

GELU (or ReLU)

↓

Dense
```

Suppose the Feed Forward Network outputs

```
[3.9,-0.05,5.9,-0.01]
```

This is again an activation.

The Transformer again performs

```
Residual Add

↓

LayerNorm
```

before sending the output to the next encoder layer.

---

# Important Observation

LayerNorm is **not waiting for a ReLU**.

It normalizes the **output of the previous layer**, regardless of whether that layer is

- Multi-Head Attention
- Dense Layer
- Feed Forward Network
- Residual Addition

Whenever you read

> "Layer Normalization normalizes the activations"

you should mentally read it as

> "Layer Normalization normalizes the output values produced by the previous layer."

# 3. Layer Normalization

Transformers do not use Batch Normalization.

Instead,

they use Layer Normalization.

Suppose one token embedding after Self-Attention is

```
[5,10,15,20]
```

LayerNorm computes

Mean

↓

Standard Deviation

↓

Normalizes only this vector.

Result

```
[-1.34,-0.45,0.45,1.34]
```

Notice

It does **not** use any other sample in the batch.

Each sample is normalized independently.

---

# Why Doesn't Transformer Use BatchNorm?

BatchNorm depends on

```
Mini Batch Statistics
```

However,

Transformers process

- variable-length sequences
- different batch sizes
- sometimes even Batch Size = 1 during inference

BatchNorm becomes inconsistent because its statistics depend on the other samples in the batch.

LayerNorm does not have this problem.

It normalizes each token independently.

Therefore,

Transformer uses Layer Normalization.

---

# Why Does Transformer Need LayerNorm?

Suppose we have

```
Embedding

↓

Transformer Layer 1

↓

Transformer Layer 2

↓

Transformer Layer 3

↓

...

↓

Transformer Layer 24
```

Every layer performs

- Multi-Head Attention
- Residual Addition
- Feed Forward Network

Without normalization,

the internal activations can gradually become too large or too small as they pass through many layers.

LayerNorm keeps these activations on a stable scale before they are passed to the next layer.

---

# Does LayerNorm Replace ReLU?

No.

Activation Functions and Normalization solve completely different problems.

Activation Function

Purpose

```
Introduce Non-Linearity
```

Examples

- ReLU
- GELU
- Sigmoid
- Tanh

Normalization

Purpose

```
Stabilize Training
```

Examples

- BatchNorm
- LayerNorm

They are not alternatives.

They work together.

---

# Where Is LayerNorm Used?

Transformer Encoder

```
Input

↓

Multi-Head Attention

↓

Residual Add

↓

Layer Normalization

↓

Feed Forward Network

↓

Residual Add

↓

Layer Normalization
```

The same structure is repeated for every encoder block.

---

# Can Transformers Be Used Only For NLP?

No.

Transformers are a general neural network architecture.

They can solve

- NLP
- Image Classification
- Object Detection
- Speech Recognition
- Time-Series Forecasting
- Regression
- Classification

Regardless of the task,

the Transformer block still contains Layer Normalization.

Only the

- Input Representation
- Output Layer
- Loss Function

change.

The Transformer architecture remains almost the same.

---

# Comparison

| Feature | Data Standardization | Batch Normalization | Layer Normalization |
|----------|----------------------|---------------------|---------------------|
| Applied Before Training | ✅ | ❌ | ❌ |
| Applied During Training | ❌ | ✅ | ✅ |
| Operates On | Input Dataset | Hidden Activations | Hidden Activations |
| Statistics Computed From | Entire Dataset | Mini Batch | One Sample |
| Used In | ML Preprocessing | ANN / CNN | Transformers, Modern Sequence Models |
| Depends On Batch Size | ❌ | ✅ | ❌ |

---

# Complete Learning Pipeline

## Traditional Machine Learning

```
Dataset

↓

StandardScaler / MinMaxScaler

↓

Machine Learning Algorithm
```

---

## ANN / CNN

```
Dataset

↓

StandardScaler (Optional)

↓

Dense / Conv

↓

BatchNorm

↓

ReLU

↓

Next Layer
```

---

## Transformer (NLP)

```
Sentence

↓

Tokenization

↓

Embedding

↓

Positional Encoding

↓

Multi-Head Attention

↓

Residual Add

↓

Layer Normalization

↓

Feed Forward

↓

Residual Add

↓

Layer Normalization
```

---

## Transformer (Regression)

```
Age
Salary
Rooms

↓

StandardScaler

↓

Linear Projection / Embedding

↓

Transformer

↓

LayerNorm

↓

Regression Output
```

Notice

Standardization is applied because the input contains numerical features.

LayerNorm is still applied because it is part of the Transformer architecture.

---

# Key Takeaways

- **Standardization / Min-Max Normalization** prepares the input data before training.
- **Batch Normalization** stabilizes hidden activations using statistics computed from the mini-batch.
- **Layer Normalization** stabilizes hidden activations independently for each sample.
- Words are **not standardized** before embedding because they are categorical, not numerical.
- Transformers use **Layer Normalization** because it works independently of batch size and sequence length.
- Layer Normalization is part of the Transformer architecture and is used regardless of whether the task is NLP, regression, classification, vision, or time-series.
