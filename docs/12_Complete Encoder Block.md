# Complete Encoder Block

## Introduction

In the previous chapters, we learned every component of the Encoder block individually.

- Multi-Head Self-Attention
- Residual (Skip) Connection
- Layer Normalization
- Feed Forward Network (FFN)
- Residual (Skip) Connection
- Layer Normalization

Now, let's combine everything to understand how a complete Encoder block works.

---

# Complete Encoder Block

```text
Input Embeddings

↓

Positional Encoding

↓

Multi-Head Self-Attention

↓

Residual Connection

↓

Layer Normalization

↓

Feed Forward Network

↓

Residual Connection

↓

Layer Normalization

↓

Encoder Block Output
```

The output after the second Layer Normalization becomes the final output of **one Encoder block**.

---

# Is the Transformer Made of Only One Encoder Block?

No.

The original Transformer is built by **stacking multiple Encoder blocks** one after another.

```text
Encoder Block 1

↓

Encoder Block 2

↓

Encoder Block 3

↓

...

↓

Encoder Block N
```

Each Encoder block has exactly the same architecture.

Only the learned weights are different.

---

# How Many Encoder Blocks Does a Transformer Have?

The original **Attention Is All You Need** paper uses

```text
6 Encoder Blocks
```

```text
Input

↓

Encoder 1

↓

Encoder 2

↓

Encoder 3

↓

Encoder 4

↓

Encoder 5

↓

Encoder 6

↓

Final Encoder Output
```

Each Encoder receives the output of the previous Encoder.

---

# Can Transformers Have More Than 6 Encoder Blocks?

Yes.

The number of Encoder blocks is a **hyperparameter** chosen by the model designer.

Some examples are

| Model | Number of Encoder Blocks |
|--------|-------------------------:|
| Original Transformer | 6 |
| BERT Base | 12 |
| BERT Large | 24 |
| ViT Base | 12 |
| ViT Large | 24 |

Larger models usually use more Encoder blocks because they can learn more complex representations.

---

# What Does Each Encoder Learn?

Although every Encoder block has the same architecture, they do not learn the same information.

For example

Encoder 1

- learns simple relationships
- nearby words
- local context

Encoder 3

- learns phrase-level meaning
- grammatical relationships

Encoder 6

- learns deeper semantic meaning
- long-range dependencies
- sentence-level understanding

As the information moves through multiple Encoder blocks, the representation becomes richer and more meaningful.

---

# How Is the Encoder Output Passed to the Decoder?

After the final Encoder block finishes, its output is passed directly to the Decoder.

```text
Encoder

↓

Final Encoder Output

↓

Decoder
```

Inside the Decoder, this output is used by the **Encoder–Decoder Attention** layer.

The Decoder uses this information to understand the input sentence while generating the output sentence.

For example

```text
English

I love cats.
```

After passing through all Encoder blocks,

```text
Rich Context Representation
```

This representation is sent to the Decoder, which generates

```text
French

J'aime les chats.
```

The Decoder attends to the Encoder output while predicting each word.

---

# Encoder and Decoder Together

```text
Input Sentence

↓

Embedding

↓

Positional Encoding

↓

Encoder Block 1

↓

Encoder Block 2

↓

...

↓

Encoder Block N

↓

Final Encoder Output

↓

Decoder Block 1

↓

Decoder Block 2

↓

...

↓

Decoder Block N

↓

Linear Layer

↓

Softmax

↓

Predicted Word
```

---

# Key Takeaways

- One Encoder block consists of Multi-Head Attention, FFN, Residual Connections, and Layer Normalization.
- The output of one Encoder becomes the input to the next Encoder.
- The original Transformer uses **6 Encoder blocks**.
- Modern Transformer models may use more or fewer Encoder blocks depending on the architecture.
- The output of the final Encoder block is passed to the Decoder through the Encoder–Decoder Attention layer.
