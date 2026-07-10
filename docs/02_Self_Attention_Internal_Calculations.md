# Self-Attention Internal Calculations

> This document demonstrates the complete forward pass of a **single
> Self-Attention layer** using a simple numerical example. It includes
> tensor shapes, matrix multiplications, attention score calculation,
> scaling, Softmax, and weighted sum.

------------------------------------------------------------------------

You: i got basic undersatnding of the self attention now take an real
time example where real compines handlies and show me how the internal
calulation happns along with what the dimensions of the tensors
embeddings shape and every calulation PyTutor: Excellent! It sounds like
you're ready to get your hands dirty with some real numbers. This is
where Self-Attention truly clicks into place.

We're going to take a very small, simple sentence and manually walk
through every single calculation. We'll track the dimensions of all our
tensors and explain why each step is performed.

------------------------------------------------------------------------

### **Scenario: Understanding "I love cats"**

Imagine a company that wants to analyze short user comments to
understand sentiment or topic. A simple example like "I love cats" might
be processed by their NLP system. We'll use this sentence to see how
Self-Attention works internally to create a richer representation of
each word.

For our numerical example, we'll keep the dimensions very small so the
calculations are manageable, but the principles scale up to much larger
models.

Let's define our key parameters: \* **Sequence Length (`seq_len`)**: The
number of words in our input sentence. \* **Embedding Dimension
(`d_model`)**: The size of each word embedding.

For our sentence "I love cats": \* `seq_len = 3` (for "I", "love",
"cats") \* Let's choose `d_model = 4`. In real models, this is often
512, 768, or more.

------------------------------------------------------------------------

### **1. Input: Sentence → Tokens → Embeddings**

First, our input sentence needs to be converted into numerical
representations that a neural network can understand.

-   **Sentence:** "I love cats"
-   **Tokens:** \["I", "love", "cats"\] (Each word is a token)

Next, each token is converted into a vector called an **embedding**.
These embeddings are typically learned during training and capture the
semantic meaning of words.

Let's assume we have the following (simplified, dummy) embeddings for
our words. Each embedding is a row vector of size `d_model`.

    # Dummy Embeddings (d_model = 4)
    # Each row represents a word's embedding

  Word       Embedding Vector `(1 x d_model)`
  ---------- ----------------------------------
  **I**      `[0.1, 0.2, 0.3, 0.4]`
  **love**   `[0.5, 0.6, 0.7, 0.8]`
  **cats**   `[0.9, 0.0, 0.1, 0.2]`

We can stack these individual word embeddings into a single **Embedding
Matrix (X)**.

    # Embedding Matrix (X)
    # Shape: (seq_len, d_model) = (3, 4)

X = \[ 0.1 0.2 0.3 0.4 \\ 0.5 0.6 0.7 0.8 \\ 0.9 0.0 0.1 0.2\]

-   **Dimensions of X:** `(3, 4)`
    -   3 rows: one for each word/token ("I", "love", "cats")
    -   4 columns: the dimension of each word's embedding

------------------------------------------------------------------------

### **2. Query, Key, and Value (Q, K, V)**

This is a crucial step in Self-Attention. Instead of directly using the
word embeddings, we transform them into three different representations:
Query (Q), Key (K), and Value (V).

**Why do we need Query, Key, and Value?**

-   **Problem without them:** If we just used the raw embeddings to
    calculate attention, it would be like asking "How much does 'I'
    relate to 'love'?" using the same representation for both roles.
    This isn't flexible enough.
-   **What problem they solve:** By having three distinct roles (Query,
    Key, Value), we can make the attention mechanism much more powerful
    and nuanced:
    -   **Query (Q):** Represents "What am I looking for?" or "What
        information am I interested in?". It's like a search query.
    -   **Key (K):** Represents "What do I have?" or "What information
        does this word offer?". It's like the index or label for a piece
        of information.
    -   **Value (V):** Represents "What information is actually stored
        here?". It's the actual content associated with the Key.
-   **Analogy:** Imagine searching for a book in a library.
    -   You (the **Query**) have a specific topic or title you're
        looking for.
    -   Each book (the **Key**) has a label or catalog entry describing
        its content.
    -   If you find a match, you pull out the actual book (the
        **Value**) to read its content.

In Self-Attention, *each word* in the sentence will act as a Query,
comparing itself to *every other word* (including itself) acting as
Keys. Based on this comparison, it will extract information from the
corresponding Values.

------------------------------------------------------------------------

### **3. Weight Matrices: WQ, WK, WV**

To transform our input embeddings (X) into Q, K, and V, we multiply X by
three different, learned weight matrices: WQ, WK, WV.

    # Weight Matrices
    # These are trainable parameters of the Self-Attention layer.
    # Their shape is typically (d_model, d_model)
    # We'll use random small numbers for demonstration.

Let `d_k = d_model` for simplicity in this example (often `d_k` is
`d_model / num_heads` in multi-head attention, but we're only doing
single-head here). So, the dimension of Q, K, and V vectors will also be
`d_model = 4`.

-   **Why are they trainable?** These matrices learn over time during
    the training process to project the original word embeddings into
    spaces where "similarity" for attention is most meaningful.
-   **Why three different matrices?** Each matrix serves a distinct
    purpose:
    -   WQ: Transforms embeddings into "Query" representations.
    -   WK: Transforms embeddings into "Key" representations.
    -   WV: Transforms embeddings into "Value" representations. If we
        used the same matrix, Q, K, and V would be identical, severely
        limiting the model's ability to learn complex relationships.

Let's define our dummy weight matrices. Each will have shape
`(d_model, d_model) = (4, 4)`.

WQ = \[ 0.1 0.0 0.2 0.1 \\ 0.0 0.1 0.1 0.0 \\ 0.1 0.2 0.0 0.1 \\ 0.0 0.1
0.1 0.2\]

WK = \[ 0.2 0.1 0.0 0.0 \\ 0.1 0.0 0.2 0.1 \\ 0.0 0.1 0.1 0.0 \\ 0.1 0.0
0.2 0.1\]

WV = \[ 0.0 0.1 0.1 0.0 \\ 0.1 0.0 0.0 0.2 \\ 0.0 0.1 0.1 0.0 \\ 0.2 0.0
0.1 0.1\]

------------------------------------------------------------------------

### **Calculation: Embeddings × Weight Matrices → Q, K, V**

Now, let's perform the matrix multiplications.

**The formula:** \* Q = X × WQ \* K = X × WK \* V = X × WV

**Dimensions:** \* `X`: `(seq_len, d_model)` = `(3, 4)` \* `WQ, WK, WV`:
`(d_model, d_model)` = `(4, 4)` \* `Q, K, V`: `(seq_len, d_model)` =
`(3, 4)`

Each row in X (an individual word embedding) will be multiplied by the
weight matrix to produce a row in Q, K, and V.

#### **Calculating Q:** Q = X × WQ

Q = \[ 0.1 0.2 0.3 0.4 \\ 0.5 0.6 0.7 0.8 \\ 0.9 0.0 0.1 0.2\] × \[ 0.1
0.0 0.2 0.1 \\ 0.0 0.1 0.1 0.0 \\ 0.1 0.2 0.0 0.1 \\ 0.0 0.1 0.1 0.2\]

Let's calculate the first row of Q (Query vector for "I"):
`[0.1, 0.2, 0.3, 0.4]` (embedding for "I") × WQ

-   Q\_{1,1} = (0.1 × 0.1) + (0.2 × 0.0) + (0.3 × 0.1) + (0.4 × 0.0) =
    0.01 + 0.0 + 0.03 + 0.0 = 0.04
-   Q\_{1,2} = (0.1 × 0.0) + (0.2 × 0.1) + (0.3 × 0.2) + (0.4 × 0.1) =
    0.0 + 0.02 + 0.06 + 0.04 = 0.12
-   Q\_{1,3} = (0.1 × 0.2) + (0.2 × 0.1) + (0.3 × 0.0) + (0.4 × 0.1) =
    0.02 + 0.02 + 0.0 + 0.04 = 0.08
-   Q\_{1,4} = (0.1 × 0.1) + (0.2 × 0.0) + (0.3 × 0.1) + (0.4 × 0.2) =
    0.01 + 0.0 + 0.03 + 0.08 = 0.12

So, the Query for "I" is `[0.04, 0.12, 0.08, 0.12]`

-   **Dimensions of Q:** `(3, 4)`

#### **Calculating K:** K = X × WK

K = \[ 0.1 0.2 0.3 0.4 \\ 0.5 0.6 0.7 0.8 \\ 0.9 0.0 0.1 0.2\] × \[ 0.2
0.1 0.0 0.0 \\ 0.1 0.0 0.2 0.1 \\ 0.0 0.1 0.1 0.0 \\ 0.1 0.0 0.2 0.1\]

Let's calculate the first row of K (Key vector for "I"):
`[0.1, 0.2, 0.3, 0.4]` (embedding for "I") × WK

-   K\_{1,1} = (0.1 × 0.2) + (0.2 × 0.1) + (0.3 × 0.0) + (0.4 × 0.1) =
    0.02 + 0.02 + 0.0 + 0.04 = 0.08
-   K\_{1,2} = (0.1 × 0.1) + (0.2 × 0.0) + (0.3 × 0.1) + (0.4 × 0.0) =
    0.01 + 0.0 + 0.03 + 0.0 = 0.04
-   K\_{1,3} = (0.1 × 0.0) + (0.2 × 0.2) + (0.3 × 0.1) + (0.4 × 0.2) =
    0.0 + 0.04 + 0.03 + 0.08 = 0.15
-   K\_{1,4} = (0.1 × 0.0) + (0.2 × 0.1) + (0.3 × 0.0) + (0.4 × 0.1) =
    0.0 + 0.02 + 0.0 + 0.04 = 0.06

So, the Key for "I" is `[0.08, 0.04, 0.15, 0.06]`

#### **Calculating V:** V = X × WV

V = \[ 0.1 0.2 0.3 0.4 \\ 0.5 0.6 0.7 0.8 \\ 0.9 0.0 0.1 0.2\] × \[ 0.0
0.1 0.1 0.0 \\ 0.1 0.0 0.0 0.2 \\ 0.0 0.1 0.1 0.0 \\ 0.2 0.0 0.1 0.1\]

Let's calculate the first row of V (Value vector for "I"):
`[0.1, 0.2, 0.3, 0.4]` (embedding for "I") × WV

-   V\_{1,1} = (0.1 × 0.0) + (0.2 × 0.1) + (0.3 × 0.0) + (0.4 × 0.2) =
    0.0 + 0.02 + 0.0 + 0.08 = 0.10
-   V\_{1,2} = (0.1 × 0.1) + (0.2 × 0.0) + (0.3 × 0.1) + (0.4 × 0.0) =
    0.01 + 0.0 + 0.03 + 0.0 = 0.04
-   V\_{1,3} = (0.1 × 0.1) + (0.2 × 0.0) + (0.3 × 0.1) + (0.4 × 0.1) =
    0.01 + 0.0 + 0.03 + 0.04 = 0.08
-   V\_{1,4} = (0.1 × 0.0) + (0.2 × 0.2) + (0.3 × 0.0) + (0.4 × 0.1) =
    0.0 + 0.04 + 0.0 + 0.04 = 0.08

So, the Value for "I" is `[0.10, 0.04, 0.08, 0.08]`

### Calculating Query (Q): Q = X × WQ

```
Q = X × WQ

X =
[
 0.1 0.2 0.3 0.4
 0.5 0.6 0.7 0.8
 0.9 0.0 0.1 0.2
]

WQ =
[
 0.1 0.0 0.2 0.1
 0.0 0.1 0.1 0.0
 0.1 0.2 0.0 0.1
 0.0 0.1 0.1 0.2
]
```

### Query Vector for "love"

Embedding for **love**

```
[0.5, 0.6, 0.7, 0.8]
```

```
Q₂₁ = (0.5 × 0.1) + (0.6 × 0.0) + (0.7 × 0.1) + (0.8 × 0.0)
     = 0.05 + 0.00 + 0.07 + 0.00
     = 0.12

Q₂₂ = (0.5 × 0.0) + (0.6 × 0.1) + (0.7 × 0.2) + (0.8 × 0.1)
     = 0.00 + 0.06 + 0.14 + 0.08
     = 0.28

Q₂₃ = (0.5 × 0.2) + (0.6 × 0.1) + (0.7 × 0.0) + (0.8 × 0.1)
     = 0.10 + 0.06 + 0.00 + 0.08
     = 0.24

Q₂₄ = (0.5 × 0.1) + (0.6 × 0.0) + (0.7 × 0.1) + (0.8 × 0.2)
     = 0.05 + 0.00 + 0.07 + 0.16
     = 0.28
```

**Query Vector for "love"**

```
[0.12, 0.28, 0.24, 0.28]
```

---

### Query Vector for "cats"

Embedding for **cats**

```
[0.9, 0.0, 0.1, 0.2]
```

```
Q₃₁ = (0.9 × 0.1) + (0.0 × 0.0) + (0.1 × 0.1) + (0.2 × 0.0)
     = 0.09 + 0.00 + 0.01 + 0.00
     = 0.10

Q₃₂ = (0.9 × 0.0) + (0.0 × 0.1) + (0.1 × 0.2) + (0.2 × 0.1)
     = 0.00 + 0.00 + 0.02 + 0.02
     = 0.04

Q₃₃ = (0.9 × 0.2) + (0.0 × 0.1) + (0.1 × 0.0) + (0.2 × 0.1)
     = 0.18 + 0.00 + 0.00 + 0.02
     = 0.20

Q₃₄ = (0.9 × 0.1) + (0.0 × 0.0) + (0.1 × 0.1) + (0.2 × 0.2)
     = 0.09 + 0.00 + 0.01 + 0.04
     = 0.14
```

**Query Vector for "cats"**

```
[0.10, 0.04, 0.20, 0.14]
```

---

## Calculating Key (K): K = X × WK

### Key Vector for "love"

Embedding

```
[0.5, 0.6, 0.7, 0.8]
```

```
K₂₁ = 0.24
K₂₂ = 0.12
K₂₃ = 0.35
K₂₄ = 0.14
```

**Key Vector**

```
[0.24, 0.12, 0.35, 0.14]
```

---

### Key Vector for "cats"

Embedding

```
[0.9, 0.0, 0.1, 0.2]
```

```
K₃₁ = 0.20
K₃₂ = 0.10
K₃₃ = 0.05
K₃₄ = 0.02
```

**Key Vector**

```
[0.20, 0.10, 0.05, 0.02]
```

---

## Calculating Value (V): V = X × WV

### Value Vector for "love"

Embedding

```
[0.5, 0.6, 0.7, 0.8]
```

```
V₂₁ = 0.22
V₂₂ = 0.12
V₂₃ = 0.20
V₂₄ = 0.20
```

**Value Vector**

```
[0.22, 0.12, 0.20, 0.20]
```

---

### Value Vector for "cats"

Embedding

```
[0.9, 0.0, 0.1, 0.2]
```

```
V₃₁ = 0.04
V₃₂ = 0.10
V₃₃ = 0.12
V₃₄ = 0.02
```

**Value Vector**

```
[0.04, 0.10, 0.12, 0.02]
```

---

# Final Summary

| Word | Embedding | Query (Q) | Key (K) | Value (V) |
|------|-----------|-----------|---------|-----------|
| **I** | [0.1, 0.2, 0.3, 0.4] | [0.04, 0.12, 0.08, 0.12] | [0.08, 0.04, 0.15, 0.06] | [0.10, 0.04, 0.08, 0.08] |
| **love** | [0.5, 0.6, 0.7, 0.8] | [0.12, 0.28, 0.24, 0.28] | [0.24, 0.12, 0.35, 0.14] | [0.22, 0.12, 0.20, 0.20] |
| **cats** | [0.9, 0.0, 0.1, 0.2] | [0.10, 0.04, 0.20, 0.14] | [0.20, 0.10, 0.05, 0.02] | [0.04, 0.10, 0.12, 0.02] |

------------------------------------------------------------------------

### **4. Attention Score: Q × Kᵀ**

This is where the magic of "attention" happens. For each word (as a
Query), we calculate how much it relates to every other word (as a Key).

**Why dot product (Q × Kᵀ) is used:**

-  # 4. Why Do We Compute Q × Kᵀ?

At this stage, every word has been transformed into three vectors:

- Query (Q)
- Key (K)
- Value (V)

For our example:

| Word | Query (Q) | Key (K) | Value (V) |
|------|-----------|----------|-----------|
| **I** | [0.04, 0.12, 0.08, 0.12] | [0.08, 0.04, 0.15, 0.06] | [0.10, 0.04, 0.08, 0.08] |
| **love** | [0.12, 0.28, 0.24, 0.28] | [0.24, 0.12, 0.35, 0.14] | [0.22, 0.12, 0.20, 0.20] |
| **cats** | [0.10, 0.04, 0.20, 0.14] | [0.20, 0.10, 0.05, 0.02] | [0.04, 0.10, 0.12, 0.02] |

At this point, we have generated Q, K, and V for every word.

However, the model still does **not** know:

- Which words are important?
- Which words should influence each other?
- How much attention should one word pay to another?

To answer these questions, we compare the **Query** vector of every word with the **Key** vectors of all the words in the sequence.

---

## Example

Suppose we are updating the word **"love"**.

The Query vector of **love** is

```
Q(love)

[0.12, 0.28, 0.24, 0.28]
```

We compare it with every Key vector.

```
Q(love) • K(I)

↓

How related is "love" to "I"?
```

```
Q(love) • K(love)

↓

How related is "love" to itself?
```

```
Q(love) • K(cats)

↓

How related is "love" to "cats"?
```

The dot product produces one similarity score for each comparison.

A larger score means the two vectors are more similar.

A smaller score means they are less related.

---

## Why is the Key Matrix Transposed?

The Query matrix has shape

```
Q = (3 × 4)
```

The Key matrix also has shape

```
K = (3 × 4)
```

If we try to multiply

```
Q × K
```

the multiplication is **not valid**.

```
(3 × 4)

×

(3 × 4)

❌ Invalid
```

The inner dimensions do not match.

To make matrix multiplication possible, we transpose the Key matrix.

```
Kᵀ = (4 × 3)
```

Now the multiplication becomes

```
Q × Kᵀ

(3 × 4)

×

(4 × 3)

↓

(3 × 3)
```

This multiplication is valid.

---

## What Does the (3 × 3) Matrix Represent?

The resulting matrix contains the attention score between every pair of words.

```
                Key

           I    love   cats

Query I

Query love

Query cats
```

Each row represents the **Query** of one word.

Each column represents the **Key** of one word.

Every cell answers the question:

> **How much should this word pay attention to that word?**

For example,

```
[
0.12 0.84 0.04

0.21 0.70 0.09

0.18 0.15 0.67
]
```

### Row 1

The word **"I"** attends

- 12% to **I**
- 84% to **love**
- 4% to **cats**

### Row 2

The word **"love"** attends

- 21% to **I**
- 70% to **love**
- 9% to **cats**

### Row 3

The word **"cats"** attends

- 18% to **I**
- 15% to **love**
- 67% to **cats**

These values indicate the importance of every word with respect to every other word.

---

## Why Do We Need These Attention Scores?

Before Self-Attention, each word only had its own embedding.

For example,

```
love

↓

Only the embedding of "love"
```

After computing the attention scores, the word **love** can gather information from every word in the sentence.

```
love

↓

I

love

cats
```

The attention scores tell the model **how much information should be collected from each word**.

These scores are then converted into probabilities using **Softmax**.

Finally, the attention weights are multiplied with the **Value (V)** vectors to generate the new context-aware embedding.

# 4. Attention Score: Q × Kᵀ

This is where the model determines **how much each word should pay attention to every other word**.

The attention scores are calculated by taking the **dot product** between the **Query (Q)** vector of each word and the **Key (K)** vectors of all words.

---

## Query Matrix (Q)

```
Q =
[
 0.04  0.12  0.08  0.12
 0.12  0.28  0.24  0.28
 0.10  0.04  0.20  0.14
]

Shape = (3 × 4)
```

---

## Key Matrix (K)

```
K =
[
 0.08  0.04  0.15  0.06
 0.24  0.12  0.35  0.14
 0.20  0.10  0.05  0.02
]

Shape = (3 × 4)
```

---

## Transpose the Key Matrix

To perform matrix multiplication, we first transpose the Key matrix.

```
Kᵀ =
[
 0.08  0.24  0.20
 0.04  0.12  0.10
 0.15  0.35  0.05
 0.06  0.14  0.02
]

Shape = (4 × 3)
```

Now,

```
Q × Kᵀ

(3 × 4)

×

(4 × 3)

↓

(3 × 3)
```

The resulting matrix contains the **Attention Scores**.

---

# Step 1: Query of "I"

Query Vector

```
Q(I)

[0.04, 0.12, 0.08, 0.12]
```

### Score(I, I)

```
= (0.04 × 0.08)
+ (0.12 × 0.04)
+ (0.08 × 0.15)
+ (0.12 × 0.06)

= 0.0032
+ 0.0048
+ 0.0120
+ 0.0072

= 0.0272
```

---

### Score(I, love)

```
= (0.04 × 0.24)
+ (0.12 × 0.12)
+ (0.08 × 0.35)
+ (0.12 × 0.14)

= 0.0096
+ 0.0144
+ 0.0280
+ 0.0168

= 0.0688
```

---

### Score(I, cats)

```
= (0.04 × 0.20)
+ (0.12 × 0.10)
+ (0.08 × 0.05)
+ (0.12 × 0.02)

= 0.0080
+ 0.0120
+ 0.0040
+ 0.0024

= 0.0264
```

---

# Step 2: Query of "love"

Query Vector

```
Q(love)

[0.12, 0.28, 0.24, 0.28]
```

### Score(love, I)

```
= (0.12 × 0.08)
+ (0.28 × 0.04)
+ (0.24 × 0.15)
+ (0.28 × 0.06)

= 0.0096
+ 0.0112
+ 0.0360
+ 0.0168

= 0.0736
```

---

### Score(love, love)

```
= (0.12 × 0.24)
+ (0.28 × 0.12)
+ (0.24 × 0.35)
+ (0.28 × 0.14)

= 0.0288
+ 0.0336
+ 0.0840
+ 0.0392

= 0.1856
```

---

### Score(love, cats)

```
= (0.12 × 0.20)
+ (0.28 × 0.10)
+ (0.24 × 0.05)
+ (0.28 × 0.02)

= 0.0240
+ 0.0280
+ 0.0120
+ 0.0056

= 0.0696
```

---

# Step 3: Query of "cats"

Query Vector

```
Q(cats)

[0.10, 0.04, 0.20, 0.14]
```

### Score(cats, I)

```
= (0.10 × 0.08)
+ (0.04 × 0.04)
+ (0.20 × 0.15)
+ (0.14 × 0.06)

= 0.0080
+ 0.0016
+ 0.0300
+ 0.0084

= 0.0480
```

---

### Score(cats, love)

```
= (0.10 × 0.24)
+ (0.04 × 0.12)
+ (0.20 × 0.35)
+ (0.14 × 0.14)

= 0.0240
+ 0.0048
+ 0.0700
+ 0.0196

= 0.1184
```

---

### Score(cats, cats)

```
= (0.10 × 0.20)
+ (0.04 × 0.10)
+ (0.20 × 0.05)
+ (0.14 × 0.02)

= 0.0200
+ 0.0040
+ 0.0100
+ 0.0028

= 0.0368
```

---

# Final Attention Score Matrix

```
Attention Scores =
[
 0.0272  0.0688  0.0264
 0.0736  0.1856  0.0696
 0.0480  0.1184  0.0368
]

Shape = (3 × 3)
```

## Interpretation

Each **row** represents the **Query** of one word.

Each **column** represents the **Key** of one word.

| Query ↓ / Key → | I | love | cats |
|-----------------|------:|------:|------:|
| **I** | 0.0272 | 0.0688 | 0.0264 |
| **love** | 0.0736 | 0.1856 | 0.0696 |
| **cats** | 0.0480 | 0.1184 | 0.0368 |

For example:

- **0.0272** is the similarity between **Query(I)** and **Key(I)**.
- **0.0688** is the similarity between **Query(I)** and **Key(love)**.
- **0.0264** is the similarity between **Query(I)** and **Key(cats)**.

These values are called **raw attention scores**.

In the next step, these scores are divided by **√dk** and then passed through the **Softmax** function to obtain the final **attention weights**.

## Interpretation of the Raw Attention Scores

The matrix above contains the **raw similarity scores** between the Query vector of each word and the Key vectors of all words.

For example,

- **0.0272** measures the similarity between **Query(I)** and **Key(I)**.
- **0.0688** measures the similarity between **Query(I)** and **Key(love)**.
- **0.0264** measures the similarity between **Query(I)** and **Key(cats)**.

Similarly,

- **0.0736** measures the similarity between **Query(love)** and **Key(I)**.
- **0.1856** measures the similarity between **Query(love)** and **Key(love)**.
- **0.0696** measures the similarity between **Query(love)** and **Key(cats)**.

Likewise,

- **0.0480** measures the similarity between **Query(cats)** and **Key(I)**.
- **0.1184** measures the similarity between **Query(cats)** and **Key(love)**.
- **0.0368** measures the similarity between **Query(cats)** and **Key(cats)**.

At this stage, these values are only **raw similarity scores**.

They are **not probabilities** and **do not yet indicate how much attention each word receives**.

To convert these similarity scores into meaningful attention weights, the next step is:

1. Scale the scores by dividing by √dk.
2. Apply the Softmax function.

After Softmax, every row becomes a probability distribution whose values sum to 1.


------------------------------------------------------------------------

### **5. Scaling: Divide by √dk**

After calculating the raw attention scores, we divide them by the square
root of the dimension of the Key vectors (`d_k`). In our case,
`d_k = 4`.

**Why divide by √dk?**

-   **What problem it solves:** The dot product of two vectors can grow
    quite large in magnitude as the dimension of the vectors (`d_k`)
    increases. If these values become too large, they can push the
    softmax function (which comes next) into regions where its gradients
    are extremely small (saturate). This makes it difficult for the
    model to learn effectively during training.
-   **What happens without it:** Without scaling, the attention scores
    could become very large, leading to very sharp softmax distributions
    (one value close to 1, others close to 0). This means the model
    becomes too "confident" in attending to just one word, hindering its
    ability to explore more nuanced relationships and learn from subtle
    differences in similarity. It essentially makes the training
    unstable.
-   **Intuition:** Scaling helps to smooth out the attention
    distribution, preventing the softmax from becoming too extreme and
    allowing for more stable training.

After calculating the **Raw Attention Scores**, the next step is to scale them by dividing each score by the square root of the Key vector dimension.

The formula is

```
Scaled Attention Scores = Attention Scores / √dk
```

where

```
dk = Dimension of the Key vector
```

In our example,

```
Key Vector Dimension = 4

√dk = √4 = 2
```

Therefore,

```
Scaled Attention Scores = Attention Scores / 2
```

---

# Why Do We Need Scaling?

To understand why scaling is required, let's first understand what happens without it.

Suppose we have two vectors.

```
Q = [1, 2]

K = [3, 4]
```

Their dot product is

```
1×3 + 2×4

= 3 + 8

= 11
```

Now suppose the vectors become much larger.

```
Q = 768 values

K = 768 values
```

Every multiplication contributes to the final dot product.

Instead of

```
11
```

the score may become

```
35

62

105

180

250
```

As the vector dimension increases, the dot product naturally becomes larger.

This is exactly what happens in real Transformer models.

For example,

BERT uses

```
dk = 64
```

GPT-2 uses

```
dk = 64
```

Larger Transformer models may also use much larger dimensions.

Therefore, the raw attention scores can become very large.

---

# What Problem Does This Create?

The next operation after calculating the attention scores is **Softmax**.

Softmax converts scores into probabilities.

Consider these scores.

```
[2, 3, 4]
```

Softmax produces

```
[0.09, 0.24, 0.67]
```

The model still considers all three words.

Now imagine much larger scores.

```
[20, 30, 40]
```

Softmax becomes approximately

```
[0.00, 0.00, 1.00]
```

Now almost all the attention goes to a single word.

The remaining words receive almost zero attention.

---

# Why Is This Bad?

Suppose our sentence is

```
I love cats
```

If the attention becomes

```
[0.00, 0.00, 1.00]
```

the model completely ignores

```
I

love
```

and only looks at

```
cats
```

The model becomes **too confident**.

It cannot learn balanced relationships between words.

---

# Another Problem During Training

Deep learning models learn using **Backpropagation**.

Backpropagation depends on gradients.

When Softmax becomes

```
[0.00, 0.00, 1.00]
```

the gradients become extremely small.

This is called **gradient saturation**.

Small gradients mean

- slower learning
- unstable training
- poor convergence

As a result, the model becomes difficult to train.

---

# How Scaling Solves This Problem

Instead of directly using

```
Attention Scores
```

we divide them by

```
√dk
```

This reduces their magnitude.

Example

Raw Attention Scores

```
[
0.0272  0.0688  0.0264

0.0736  0.1856  0.0696

0.0480  0.1184  0.0368
]
```

Since

```
dk = 4

√dk = 2
```

Divide every element by 2.

```
Scaled Attention Scores

[
0.0136  0.0344  0.0132

0.0368  0.0928  0.0348

0.0240  0.0592  0.0184
]
```

Shape

```
(3 × 3)
```

Notice that every value becomes smaller.

These smaller values allow Softmax to produce a smoother probability distribution.

---

# Real-Time Example

Imagine three students writing an exam.

Without scaling

```
Marks

98

12

10
```

The topper completely dominates.

Now suppose we normalize the marks.

```
49

6

5
```

The differences become smaller.

Everyone still contributes.

Scaling works in a similar way.

It prevents one word from dominating all the attention.

---

# Python Implementation

```python
import numpy as np

attention_scores = np.array([
    [0.0272, 0.0688, 0.0264],
    [0.0736, 0.1856, 0.0696],
    [0.0480, 0.1184, 0.0368]
])

dk = 4

scaled_scores = attention_scores / np.sqrt(dk)

print(scaled_scores)
```

Output

```python
[
 [0.0136 0.0344 0.0132]
 [0.0368 0.0928 0.0348]
 [0.0240 0.0592 0.0184]
]
```

---

# Summary

- The dot product becomes larger as the vector dimension increases.
- Large attention scores make the Softmax function extremely sharp.
- Sharp Softmax distributions cause gradient saturation during training.
- Gradient saturation makes learning unstable and slow.
- Dividing by √dk keeps the attention scores at a reasonable scale.
- This allows Softmax to produce smoother attention weights and improves training stability.

-   **Dimensions of Scaled_Attention_Scores:** `(3, 3)`

------------------------------------------------------------------------

### **6. Softmax**

The scaled attention scores are still just raw numbers. We need to
convert them into a probability distribution. This is where the Softmax
function comes in.

**Why Softmax is needed:**

-   **What problem it solves:** The raw attention scores can be any real
    number (positive, negative, or zero). We want them to represent
    *weights* or *probabilities* that sum up to 1 for each query,
    indicating how much attention each word should pay to other words.
-   **What happens without it:** Without Softmax, the scores wouldn't be
    normalized, and we wouldn't have a clear, interpretable way to
    combine the Value vectors in the next step. The concept of
    "attention" (selectively weighting information) relies on these
    probabilistic weights.
-   **How it works:** # 6. Softmax: Convert Scores into Attention Weights

After scaling the attention scores, we apply the **Softmax** function.

The purpose of Softmax is to convert the raw scores into probabilities.

The attention weights satisfy two important properties:

- Every value lies between **0 and 1**.
- The values in each row sum to **1**.

The Softmax formula is

```
                ezi
Softmax(zi) =  ----------
              Σ ezj
```

where

- `zi` is the current score.
- `Σ ezj` is the sum of exponentials of all the scores in that row.

---

# Scaled Attention Scores

```
[
0.0136  0.0344  0.0132
0.0368  0.0928  0.0348
0.0240  0.0592  0.0184
]

Shape = (3 × 3)
```

---

# Step 1: Softmax for the First Row ("I")

Scaled Scores

```
[0.0136, 0.0344, 0.0132]
```

### Calculate the exponential of each value

```
e^0.0136 ≈ 1.01369

e^0.0344 ≈ 1.03500

e^0.0132 ≈ 1.01329
```

### Sum of exponentials

```
1.01369 + 1.03500 + 1.01329

= 3.06198
```

### Divide each exponential by the sum

```
0.0136

↓

1.01369 / 3.06198

= 0.3310
```

```
0.0344

↓

1.03500 / 3.06198

= 0.3380
```

```
0.0132

↓

1.01329 / 3.06198

= 0.3310
```

Therefore,

```
Attention Weights for "I"

[0.3310, 0.3380, 0.3310]
```

Notice that

```
0.3310 + 0.3380 + 0.3310

≈ 1.00
```

---

# Step 2: Softmax for the Second Row ("love")

Scaled Scores

```
[0.0368, 0.0928, 0.0348]
```

### Calculate exponentials

```
e^0.0368 ≈ 1.03749

e^0.0928 ≈ 1.09725

e^0.0348 ≈ 1.03541
```

### Sum

```
1.03749 + 1.09725 + 1.03541

= 3.17015
```

### Divide by the sum

```
1.03749 / 3.17015 = 0.3273

1.09725 / 3.17015 = 0.3461

1.03541 / 3.17015 = 0.3266
```

Therefore,

```
Attention Weights for "love"

[0.3273, 0.3461, 0.3266]
```

---

# Step 3: Softmax for the Third Row ("cats")

Scaled Scores

```
[0.0240, 0.0592, 0.0184]
```

### Calculate exponentials

```
e^0.0240 ≈ 1.02429

e^0.0592 ≈ 1.06099

e^0.0184 ≈ 1.01857
```

### Sum

```
1.02429 + 1.06099 + 1.01857

= 3.10385
```

### Divide by the sum

```
1.02429 / 3.10385 = 0.3300

1.06099 / 3.10385 = 0.3418

1.01857 / 3.10385 = 0.3282
```

Therefore,

```
Attention Weights for "cats"

[0.3300, 0.3418, 0.3282]
```

---

# Final Attention Weight Matrix

```
Attention Weights =
[
0.3310  0.3380  0.3310
0.3273  0.3461  0.3266
0.3300  0.3418  0.3282
]

Shape = (3 × 3)
```

---

# Interpretation

Each row tells us how a word distributes its attention across all the words in the sentence.

| Query ↓ / Key → | I | love | cats |
|-----------------|------:|------:|------:|
| **I** | 0.3310 | 0.3380 | 0.3310 |
| **love** | 0.3273 | 0.3461 | 0.3266 |
| **cats** | 0.3300 | 0.3418 | 0.3282 |

For example, the second row

```
[0.3273, 0.3461, 0.3266]
```

means that while updating the word **"love"**, the model gathers:

- **32.73%** of its information from **"I"**
- **34.61%** from **"love"**
- **32.66%** from **"cats"**

These attention weights are then multiplied with the **Value (V)** vectors to produce the final context-aware embeddings.

---

# Python Implementation

```python
import numpy as np

scaled_scores = np.array([
    [0.0136, 0.0344, 0.0132],
    [0.0368, 0.0928, 0.0348],
    [0.0240, 0.0592, 0.0184]
])

# Apply Softmax row-wise
exp_scores = np.exp(scaled_scores)

attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

print(attention_weights)
```

### Output

```python
[[0.3310 0.3380 0.3310]
 [0.3273 0.3461 0.3266]
 [0.3300 0.3418 0.3282]]
```
One small note: because our example uses very small attention scores, the Softmax outputs are quite close to each other (around 0.33). In real Transformer models, the vectors are much larger (for example, dk = 64), so the attention weights often differ much more dramatically, allowing the model to focus strongly on the most relevant tokens.

------------------------------------------------------------------------

### **7. Weighted Sum: Attention_Weights × Value Matrix**

Finally, we use these attention weights to create a new,
"contextualized" representation for each word. We do this by taking a
weighted sum of the Value vectors.

**Why Values are multiplied instead of Keys:**

-   **What problem it solves:** The Keys (`K`) were used to *find*
    relevance. They are like indices or labels. The Values (`V`) contain
    the *actual content* or information associated with those keys. Once
    we know *how relevant* a word (Key) is to our Query, we want to
    extract *its content* (Value).
-   **What happens without it:** If we multiplied by Keys, we would be
    blending relevance indices rather than semantic content. The
    resulting vector would not represent a rich, contextualized meaning
    of the word. Value vectors are specifically designed to carry the
    information that should be propagated.

**The calculation:** We multiply the `Attention_Weights` matrix by the
`Value` matrix (V).

-   `Attention_Weights`: `(seq_len, seq_len)` = `(3, 3)`
-   `V`: `(seq_len, d_model)` = `(3, 4)`
-   `Output_Embeddings`: `(seq_len, d_model)` = `(3, 4)`

Each row in the `Output_Embeddings` will be the new, contextualized
embedding for a word.

`\text{Output_Embeddings}`{=tex} = `\text{Attention_Weights}`{=tex} × V

`\text{Output_Embeddings}`{=tex} = \[ 0.3303 0.3357 0.3340 \\ 0.3292
0.3421 0.3287 \\ 0.3306 0.3384 0.3310\] × \[ 0.10 0.04 0.08 0.08 \\ 0.21
0.12 0.19 0.20 \\ 0.08 0.10 0.11 0.02\]

Let's calculate the first row of `Output_Embeddings` (new embedding for
"I"): `[0.3303, 0.3357, 0.3340]` (attention weights for "I") × V

-   **First element of new "I" embedding:** (0.3303 × 0.10) + (0.3357 ×
    0.21) + (0.3340 × 0.08) = 0.03303 + 0.070497 + 0.02672 = 0.130247

-   **Second element of new "I" embedding:** (0.3303 × 0.04) + (0.3357 ×
    0.12) + (0.3340 × 0.10) = 0.013212 + 0.040284 + 0.03340 = 0.086896

-   **Third element of new "I" embedding:** (0.3303 × 0.08) + (0.3357 ×
    0.19) + (0.3340 × 0.11) = 0.026424 + 0.063783 + 0.03674 = 0.126947

-   **Fourth element of new "I" embedding:** (0.3303 × 0.08) + (0.3357 ×
    0.20) + (0.3340 × 0.02) = 0.026424 + 0.06714 + 0.00668 = 0.100244

So, the new contextualized embedding for "I" is approximately
`[0.1302, 0.0869, 0.1269, 0.1002]`.

Performing similar calculations for "love" and "cats":

`\text{Output_Embeddings}`{=tex} = \[ 0.1302 0.0869 0.1269 0.1002 \\
0.1352 0.0903 0.1320 0.1054 \\ 0.1293 0.0864 0.1265 0.0999\]

-   **Dimensions of Output_Embeddings:** `(3, 4)`

------------------------------------------------------------------------

### **8. Final Output: Contextualized Embeddings**

This `Output_Embeddings` matrix is the final result of our
Self-Attention layer.

-   Each row is a new embedding for the corresponding word ("I", "love",
    "cats").
-   The shape is `(seq_len, d_model)`, which is `(3, 4)`. This is the
    same shape as the input embedding matrix `X`.

**Why the new embedding is better than the original embedding:**

-   The original embeddings (X) were static and represented the word's
    meaning in isolation, or a general context.
-   The **new, contextualized embeddings** have been enriched by
    information from all other words in the sentence, weighted by their
    relevance. For example, the new embedding for "love" in "I love
    cats" now implicitly contains information about "I" and "cats",
    based on how much "love" attends to them.
-   This means the word "love" now has a specific representation that
    reflects its meaning *in this particular sentence*, rather than its
    general meaning. If the sentence was "I love *to run*", the new
    embedding for "love" would be different, reflecting the influence of
    "to run".

This ability to create context-aware representations is the core power
of Self-Attention. These contextualized embeddings are then passed to
the next layer in the neural network for further processing (e.g.,
classification, sequence generation, etc.).

------------------------------------------------------------------------

### **Complete Forward Pass Summary for "I love cats"**

Let's visualize the entire pipeline with the tensor dimensions:

1.  **Input Sentence:** "I love cats"

    -   *(Human readable)*

2.  **Embeddings (X):**

    -   Transforms tokens into initial numerical representations.
    -   **Shape:** `(3, 4)` (seq_len, d_model)

    X = \[ 0.1 0.2 0.3 0.4 \\ 0.5 0.6 0.7 0.8 \\ 0.9 0.0 0.1 0.2\]

3.  **Linear Transformations to Q, K, V:**

    -   Q = X × WQ, K = X × WK, V = X × WV
    -   WQ, WK, WV: `(4, 4)` (d_model, d_model)
    -   **Shapes:**
        -   `Q`: `(3, 4)`
        -   `K`: `(3, 4)`
        -   `V`: `(3, 4)`

    Q = \[ `\dots`{=tex}\], K = \[ `\dots`{=tex}\], V = \[
    `\dots`{=tex}\]

4.  **Attention Score (Dot Product):**

    -   `\text{Scores}`{=tex} = Q × Kᵀ
    -   **Shape:** `(3, 3)` (seq_len, seq_len)

    `\text{Scores}`{=tex} = \[ 0.0272 0.0596 0.0496 \\ 0.0594 0.1287
    0.1083 \\ 0.0468 0.0964 0.0886\]

5.  **Scaling:**

    -   `\text{Scaled\_Scores}`{=tex} =
        `\frac{\text{Scores}}{√dk}`{=tex} (where √dk = `\sqrt{4}`{=tex}
        = 2)
    -   **Shape:** `(3, 3)`

    `\text{Scaled\_Scores}`{=tex} = \[ 0.0136 0.0298 0.0248 \\ 0.0297
    0.06435 0.05415 \\ 0.0234 0.0482 0.0443\]

6.  **Softmax:**

    -   `\text{Attention\_Weights}`{=tex} =
        `\text{Softmax}`{=tex}(`\text{Scaled\_Scores}`{=tex}) (applied
        row-wise)
    -   **Shape:** `(3, 3)`

    `\text{Attention\_Weights}`{=tex} = \[ 0.3303 0.3357 0.3340 \\
    0.3292 0.3421 0.3287 \\ 0.3306 0.3384 0.3310\]

7.  **Weighted Sum:**

    -   `\text{Output}`{=tex} = `\text{Attention\_Weights}`{=tex} × V
    -   **Shape:** `(3, 4)` (seq_len, d_model)

    `\text{Output}`{=tex} = \[ 0.1302 0.0869 0.1269 0.1002 \\ 0.1352
    0.0903 0.1320 0.1054 \\ 0.1293 0.0864 0.1265 0.0999\]

8.  **Output (Contextualized Embeddings):**

    -   New, context-aware representation for each word.
    -   **Shape:** `(3, 4)` (seq_len, d_model)

You've now walked through every single matrix multiplication and
operation in a Self-Attention layer! This is the fundamental mechanism
that allows words to "talk to each other" and derive context in models
like Transformers. You:
