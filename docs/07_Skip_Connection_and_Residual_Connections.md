# Skip Connection and Residual Connections in Transformers

After understanding

- Tokenization
- Word Embeddings
- Positional Encoding
- Multi-Head Attention

we now have a new question.

**What happens after Multi-Head Attention?**

Many beginners think the output of Multi-Head Attention is directly sent to the next layer.

This is **not true**.

The Transformer first performs a **Residual Connection (Skip Connection)** and only then applies **Layer Normalization**.

---

# Where Are We in the Transformer Encoder?

The flow so far is

```
Sentence

↓

Tokenization

↓

Word Embedding

↓

Positional Encoding

↓

Multi-Head Attention
```

Let's continue from here.

---

# Step 1 : Word Embeddings

Suppose our sentence is

```
I      love      cats
```

Assume

```
Embedding Dimension = 4
```

The embedding for the word

```
love
```

is

```
[5,1,2,3]
```

---

# Step 2 : Positional Encoding

Suppose the positional encoding for

```
love
```

is

```
[0.2,-0.3,0.1,0.4]
```

The Transformer performs an **element-wise addition**.

```
Word Embedding

+

Positional Encoding
```

Calculation

```
[5,1,2,3]

+

[0.2,-0.3,0.1,0.4]

=

[5.2,0.7,2.1,3.4]
```

This vector now contains

- Semantic Information
- Positional Information

This becomes the input to the Multi-Head Attention layer.

Let's call this vector

```
X
```

```
X

=

[5.2,0.7,2.1,3.4]
```

---

# Step 3 : Multi-Head Attention

The input

```
X
```

enters the Multi-Head Attention block.

Inside Multi-Head Attention,

the Transformer computes

- Query
- Key
- Value
- Attention Scores
- Context Vectors

Finally,

suppose the output becomes

```
MHA(X)

=

[1.8,2.5,3.0,1.6]
```

Notice

This vector is now **context-aware**.

The word

```
love
```

has learned information from

```
I

and

cats
```

---

# Can We Simply Use This Output?

A beginner might think

```
Output of Multi-Head Attention

↓

Next Layer
```

However,

the Transformer does **not** do this.

Instead,

it keeps the original input and combines it with the newly learned contextual information.

---

# Step 4 : Skip Connection (Residual Connection)

The Transformer performs

```
Original Input

+

Multi-Head Attention Output
```

Mathematically,

```
X + MHA(X)
```

Calculation

```
Original Input (Embedding + Positional Encoding)

[5.2,0.7,2.1,3.4]

+

Context-Aware Output

[1.8,2.5,3.0,1.6]

=

[7.0,3.2,5.1,5.0]
```

This operation is called the **Residual Connection**.

It is also commonly called the **Skip Connection** because the original input skips over the Multi-Head Attention layer and is added back to its output.

---

# Why Is It Called a Skip Connection?

Normally,

information flows like this.

```
Input

↓

Multi-Head Attention

↓

Output
```

With a Skip Connection,

the original input also travels directly to the Add operation.

```
             ┌─────────────────────────────┐
             │                             │
Input ───────┤                             ▼
             │                    Add (+)
             ▼                       ▲
     Multi-Head Attention            │
             │                       │
             └───────────────────────┘
```

The input **skips** the Multi-Head Attention layer.

Hence the name

```
Skip Connection
```

---

# Why Do We Add the Original Input?

Suppose

```
love
```

already has a good embedding.

Multi-Head Attention learns additional contextual information.

Instead of replacing the original representation,

the Transformer combines both.

```
Original Meaning

+

Context Information

↓

Better Representation
```

This allows the model to preserve the original semantic meaning while adding contextual knowledge.

---

# Why Are Residual Connections Important?

Residual Connections provide several advantages.

## 1. Preserve Original Information

Without Residual Connection

```
Embedding

↓

Attention

↓

Only Context
```

The original embedding could be lost.

With Residual Connection

```
Embedding

+

Context

↓

Updated Representation
```

Both pieces of information are preserved.

---

## 2. Better Gradient Flow

During backpropagation,

the gradient has a direct path through the Skip Connection.

This makes training very deep Transformer models much easier.

---

## 3. Enable Very Deep Transformers

Modern Transformers may contain

- 12 layers
- 24 layers
- 48 layers
- 96 layers

Residual Connections make training these deep architectures stable.

---

# What Happens Next?

After the Residual Connection,

the resulting vector

```
[7.0,3.2,5.1,5.0]
```

is **not** directly sent to the Feed Forward Network.

The Transformer first applies

```
Layer Normalization
```

Therefore,

the complete flow becomes

```
Sentence

↓

Tokenization

↓

Word Embedding

↓

Positional Encoding

↓

Embedding + Positional Encoding

↓

Multi-Head Attention

↓

Context-Aware Embeddings

↓

Residual Add (Skip Connection)

↓

Layer Normalization

↓

Feed Forward Network

↓

Residual Add

↓

Layer Normalization

↓

Next Encoder Layer
```

---

# Key Takeaways

- The input to Multi-Head Attention is **Embedding + Positional Encoding**.
- Multi-Head Attention produces **Context-Aware Embeddings**.
- The Transformer does **not discard** the original input.
- Instead, it computes

```
Output = X + MultiHeadAttention(X)
```

where

```
X = Embedding + Positional Encoding
```

- This addition is called the **Residual Connection**.
- Because the input skips over the Multi-Head Attention block, it is also called a **Skip Connection**.
- Only after this addition does the Transformer apply **Layer Normalization**.
