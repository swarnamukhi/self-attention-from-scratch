# Self-Attention: Internal Calculations

## Introduction

In this document, we will perform the complete forward pass of a single Self-Attention layer using a small numerical example.

The goal is to understand:

- How Query (Q), Key (K), and Value (V) are generated
- How attention scores are calculated
- Why scaling is required
- How Softmax converts scores into attention weights
- How the final context-aware embeddings are produced

---

# Step 1: Input Sentence

Consider the following sentence:

```
I love cats
```

The sentence contains **3 tokens**.

Each token is represented using an embedding of dimension **4**.

Therefore,

```
Embedding Matrix (X)

Shape = (3 × 4)
```

Example:

```
          e1  e2  e3  e4
I       [0.1 0.2 0.3 0.4]
love    [0.5 0.6 0.7 0.8]
cats    [0.9 1.0 1.1 1.2]
```

---

# Step 2: Generate Query, Key and Value

The embedding matrix is transformed into three different matrices.

```
Q = X × WQ

K = X × WK

V = X × WV
```

where

```
X  = (3 × 4)

WQ = (4 × 4)

WK = (4 × 4)

WV = (4 × 4)
```

After multiplication,

```
Q = (3 × 4)

K = (3 × 4)

V = (3 × 4)
```

Each token now has

- Query Vector
- Key Vector
- Value Vector

---

# Step 3: Calculate Attention Scores

Attention scores are calculated by comparing every Query vector with every Key vector.

Formula

```
Attention Scores = Q × Kᵀ
```

Since

```
Q = (3 × 4)

Kᵀ = (4 × 3)
```

the resulting matrix becomes

```
Attention Scores

Shape = (3 × 3)
```

Example

```
[
 2.4  3.8  1.2
 1.5  4.2  2.6
 0.9  2.7  5.1
]
```

Each row represents one token attending to every token in the sequence.

---

# Step 4: Scaling

Large attention scores can make Softmax produce extremely large probabilities for one token and very small probabilities for the others.

To prevent this, divide the scores by

```
√dk
```

where

```
dk = Dimension of the Key vectors
```

If

```
dk = 4
```

then

```
√4 = 2
```

Scaled Attention Scores

```
Original

[
2.4 3.8 1.2
1.5 4.2 2.6
0.9 2.7 5.1
]

↓

Scaled

[
1.2 1.9 0.6
0.75 2.1 1.3
0.45 1.35 2.55
]
```

---

# Step 5: Apply Softmax

Softmax converts every row into probabilities.

Example

```
Before

[1.2 1.9 0.6]

↓

After Softmax

[0.28 0.56 0.16]
```

Each row sums to **1**.

The complete Attention Weight Matrix becomes

```
[
0.28 0.56 0.16

0.15 0.58 0.27

0.08 0.20 0.72
]
```

---

# Step 6: Weighted Sum

The attention weights are multiplied with the Value matrix.

Formula

```
Output = Attention Weights × V
```

Shape

```
(3 × 3)

×

(3 × 4)

=

(3 × 4)
```

The resulting matrix contains the new context-aware embeddings.

---

# Step 7: Final Output

The final output has the same shape as the input embedding matrix.

```
Input Embeddings

Shape = (3 × 4)

↓

Self-Attention

↓

Output Embeddings

Shape = (3 × 4)
```

Although the shape remains the same, each output embedding now contains contextual information from every token in the sequence.

---

# Complete Forward Pass

```
Sentence

↓

Tokenization

↓

Embedding Matrix (3 × 4)

↓

Generate Q, K and V

↓

Q × Kᵀ

↓

Scaling

↓

Softmax

↓

Attention Weights

↓

Attention Weights × V

↓

Context-Aware Embeddings
```

---

# Key Takeaways

- The input to Self-Attention is the embedding matrix.
- Q, K and V are generated using trainable weight matrices.
- Attention scores measure the similarity between Query and Key vectors.
- Scaling prevents very large attention scores.
- Softmax converts attention scores into probabilities.
- The attention weights are multiplied with the Value matrix.
- The output is a context-aware embedding for every token.

---

## Status

Completed

- ✅ Input Embeddings
- ✅ Query, Key and Value Generation
- ✅ Attention Score Calculation
- ✅ Scaling
- ✅ Softmax
- ✅ Weighted Sum
- ✅ Output Embeddings
