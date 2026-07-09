# Self-Attention Theory

## Introduction

Self-Attention is a mechanism that allows every token in a sequence to attend to every other token in the same sequence. It learns how important each token is with respect to the others and generates a new context-aware representation for every token.

Unlike RNNs and LSTMs, Self-Attention does not process tokens sequentially. Instead, all tokens interact with each other simultaneously, making it efficient for capturing long-range dependencies.

---

# Why Self-Attention?

Self-Attention was introduced to overcome some of the limitations of recurrent models.

It helps to:

- Capture relationships between distant words.
- Understand the context of each token using the entire sequence.
- Process all tokens in parallel.
- Generate better contextual embeddings.

---

# Input to Self-Attention

The input to a Self-Attention layer is the **Embedding Matrix**.

Example sentence:

```
I love cats
```

Suppose each word is represented using an embedding of size **4**.

Then the embedding matrix will have the shape:

```
Embedding Matrix (X)

Shape = (3 × 4)
```

where

- **3** = Number of tokens
- **4** = Embedding dimension

---

# Query, Key and Value

Each embedding is transformed into three different vectors.

- Query (Q)
- Key (K)
- Value (V)

Each vector has a different purpose.

### Query (Q)

Represents what the current token is looking for.

### Key (K)

Represents what information each token contains.

### Value (V)

Represents the information that will be passed to the final output after attention is computed.

---

# Weight Matrices

The Query, Key and Value vectors are created using three trainable weight matrices.

- WQ
- WK
- WV

Suppose the embedding matrix is

```
X = (3 × 4)
```

Then

```
WQ = (4 × 4)

WK = (4 × 4)

WV = (4 × 4)
```

Using matrix multiplication,

```
Q = X × WQ

K = X × WK

V = X × WV
```

The resulting shapes become

```
Q = (3 × 4)

K = (3 × 4)

V = (3 × 4)
```

Although WQ, WK and WV have shape **(4 × 4)**, they simply transform every 4-dimensional embedding into another learned 4-dimensional representation.

---

# Complete Flow

```
Sentence
      │
      ▼
Tokenization
      │
      ▼
Embedding Matrix (X)
      │
      ▼
Generate Q, K and V
      │
      ▼
Attention Score
      │
      ▼
Scaling
      │
      ▼
Softmax
      │
      ▼
Weighted Sum
      │
      ▼
Context-Aware Embeddings
```

---

# Summary

- Self-Attention allows every token to interact with every other token in the same sequence.
- The input to Self-Attention is the embedding matrix.
- Every embedding is transformed into Query, Key and Value vectors.
- WQ, WK and WV are trainable weight matrices used to generate Q, K and V.
- Self-Attention produces context-aware embeddings by assigning different importance to different tokens in the sequence.

---

## Repository Status

**Completed**

- ✅ Introduction
- ✅ Why Self-Attention?
- ✅ Input Embeddings
- ✅ Query, Key and Value
- ✅ Weight Matrices (WQ, WK, WV)
- ✅ Complete Flow
- ✅ Summary

This document will be updated as I continue learning Self-Attention.
