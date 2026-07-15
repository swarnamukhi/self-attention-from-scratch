# Multi-Head Attention - Comparison with Single Head Attention

Before learning the calculations of Multi-Head Attention, let's compare it with Single Head Self-Attention.

---

# Single Head Self-Attention

Suppose the input sentence is

```
I      love      cats
```

Assume

```
Number of Tokens (n) = 3

Embedding Dimension (d_model) = 4
```

The input embedding matrix is

```
X

Shape

3 × 4
```

where

- 3 = number of tokens
- 4 = embedding dimension

---

## Step 1 : Compute Query, Key and Value

Single Head uses one set of learnable weight matrices.

```
WQ

WK

WV
```

Each matrix has shape

```
4 × 4
```

Therefore,

```
Q = X × WQ

(3×4)

×

(4×4)

=

3×4
```

Similarly,

```
K = X × WK

↓

3×4
```

```
V = X × WV

↓

3×4
```

---

## Step 2 : Attention Score

Compute

```
Q × KT
```

```
(3×4)

×

(4×3)

=

3×3
```

The result is

```
Attention Score Matrix

3×3
```

Every word now has an attention score with every other word.

---

## Step 3 : Scale

```
QKT

────────
 √dk
```

Since

```
dk = 4
```

divide every score by

```
√4 = 2
```

---

## Step 4 : Softmax

Apply Softmax row-wise.

The attention scores become probabilities.

---

## Step 5 : Context Vector

Multiply

```
Attention

×

V
```

```
(3×3)

×

(3×4)

=

3×4
```

This produces the Context Matrix.

```
Output

3×4
```

Every token now has one context vector of dimension 4.

---

# Problem with Single Head

Only **one** set of Query, Key and Value matrices is used.

```
WQ

WK

WV
```

Therefore,

only **one attention pattern** is learned.

All grammatical, semantic and contextual relationships are mixed into a single representation.

---

# Multi-Head Attention

Instead of one attention mechanism,

the Transformer runs several Self-Attention mechanisms in parallel.

Suppose

```
d_model = 4

Heads = 2
```

The embedding matrix remains

```
X

Shape

3 × 4
```

Notice

The embedding dimension is **NOT divided**.

The same input embedding is sent to every head.

---

# Step 1 : Create Multiple Projection Matrices

Instead of one

```
WQ

WK

WV
```

we create

Head 1

```
WQ₁

WK₁

WV₁
```

Head 2

```
WQ₂

WK₂

WV₂
```

Each head learns its own projection.

---

# Step 2 : Reduce Feature Dimension

Since

```
Heads = 2
```

each head works with

```
dk = d_model / Heads

=

4 / 2

=

2
```

Therefore,

instead of

```
WQ

4×4
```

Head 1 uses

```
WQ₁

4×2
```

Head 2 uses

```
WQ₂

4×2
```

Similarly,

```
WK₁

4×2
```

```
WV₁

4×2
```

and

```
WK₂

4×2
```

```
WV₂

4×2
```

---

# Step 3 : Compute Query, Key and Value

Head 1

```
Q₁

=

X × WQ₁
```

```
(3×4)

×

(4×2)

=

3×2
```

Similarly,

```
K₁

↓

3×2
```

```
V₁

↓

3×2
```

---

Head 2

```
Q₂

=

X × WQ₂
```

```
(3×4)

×

(4×2)

=

3×2
```

Similarly,

```
K₂

↓

3×2
```

```
V₂

↓

3×2
```

Notice

The number of rows (tokens) never changes.

Only the feature dimension changes.

---

# Step 4 : Self-Attention Inside Each Head

Now every head independently performs

```
Q × KT

↓

Scaling

↓

Softmax

↓

Attention × V
```

Exactly the same operations as Single Head Self-Attention.

Head 1 produces

```
Output₁

3×2
```

Head 2 produces

```
Output₂

3×2
```

---

# Step 5 : Concatenation

The outputs are concatenated along the feature dimension.

```
Head 1 Output

3×2
```

```
Head 2 Output

3×2
```

Concatenate

↓

```
3×4
```

The original embedding dimension is restored.

---

# Step 6 : Final Linear Projection

The concatenated output is multiplied by another learnable weight matrix.

```
WO
```

Shape

```
4 × 4
```

```
Output

=

Concat × WO
```

```
(3×4)

×

(4×4)

=

3×4
```

This becomes the final output of the Multi-Head Attention layer.

---

# Why Concatenate?

Suppose

Head 1 learned

- Grammar

Head 2 learned

- Semantic Relationships

Each head produces its own context vector.

Instead of discarding these independent representations,

the Transformer combines them by concatenation.

The final linear layer (`WO`) then mixes the information from all heads into a single representation.

---

# Comparison

| Property | Single Head | Multi-Head |
|------------|-------------|------------|
| Number of Attention Heads | 1 | h |
| Input Embedding | n × d_model | n × d_model |
| WQ Shape | d_model × d_model | d_model × (d_model/h) |
| WK Shape | d_model × d_model | d_model × (d_model/h) |
| WV Shape | d_model × d_model | d_model × (d_model/h) |
| Q Shape | n × d_model | n × (d_model/h) |
| K Shape | n × d_model | n × (d_model/h) |
| V Shape | n × d_model | n × (d_model/h) |
| Output per Head | n × d_model | n × (d_model/h) |
| Concatenated Output | Not Required | n × d_model |
| Final Linear Layer | Not Required | WO |

---

# Key Takeaways

- The input embedding is **not split**. Every head receives the same embedding matrix.
- Each head has its own learnable `WQ`, `WK`, and `WV`.
- These projection matrices reduce the feature dimension from `d_model` to `d_model / h`.
- Every head performs a complete Self-Attention calculation independently.
- The outputs of all heads are concatenated.
- A final linear projection (`WO`) combines information from all heads and restores the final representation.
