# 8. What is a Context-Aware Embedding?

After performing the weighted sum, we obtain a **new embedding** for every word.

For example,

Original embedding of **"I"**

```
[0.10, 0.04, 0.08, 0.08]
```

After Self-Attention

```
[0.1207, 0.0869, 0.1338, 0.1007]
```

Notice that the values have changed.

This new vector is called a **Context-Aware Embedding**.

---

# Why Did the Embedding Change?

Originally,

the embedding of **"I"** only represented the word **"I"**.

It did **not** know

- what comes after it,
- which words are nearby,
- or the meaning of the complete sentence.

After Self-Attention,

the new embedding becomes

```
New Embedding(I)

=

Information from I

+

Information from love

+

Information from cats
```

Instead of representing only **"I"**,

it now represents

```
"I" in the context of

I love cats
```

This is why it is called a **context-aware embedding**.

---

# Example

Sentence

```
I love cats
```

Original Embedding

```
"I"

↓

Only represents the word "I"
```

Context-Aware Embedding

```
"I"

↓

Represents

"I"

+

"love"

+

"cats"
```

The same happens for every word.

```
love

↓

love

+

I

+

cats
```

```
cats

↓

cats

+

I

+

love
```

Every word now understands the entire sentence.

---

# Why Is This Useful?

Suppose we want to predict the next word.

Sentence

```
I love cats
```

The model does **not** use the original embeddings.

Instead,

it uses the **context-aware embeddings**.

Why?

Because the new embeddings contain information from the entire sentence.

This gives the model much richer information for making predictions.

---

# How Does This Help in Next Word Prediction?

Suppose the sentence is

```
I love
```

The model creates context-aware embeddings.

The embedding for **"love"** now contains information from

```
I

+

love
```

This embedding is then passed through the remaining Transformer layers and finally to a Linear layer and Softmax.

The model may predict

```
cats

dogs

pizza

coding
```

The word with the highest probability becomes the next predicted word.

---

# How Does This Help in Summarization?

Suppose the document is

```
The cat was sitting on the mat.

It was sleeping peacefully.

The dog came near.

The cat woke up.
```

Every word receives information from other words.

For example,

the embedding of

```
cat
```

is updated using information from

```
sleeping

dog

mat

woke
```

As a result,

the model understands

- who is performing the action,
- what happened,
- and how different parts of the document are related.

This allows the model to generate an accurate summary instead of simply copying sentences.

---

# Why Is This Better Than the Original Embeddings?

Original Embedding

```
cat

↓

Only represents the word "cat"
```

Context-Aware Embedding

```
cat

↓

Represents

cat

+

sleeping

+

dog

+

woke

+

mat
```

The new embedding contains much richer semantic information.

---

# Where Are These New Embeddings Used?

The Context-Aware Embeddings become the input to the next layer.

```
Embeddings

↓

Self-Attention

↓

Context-Aware Embeddings

↓

Feed Forward Network

↓

Next Transformer Layer

↓

...

↓

Final Linear Layer

↓

Softmax

↓

Prediction
```

Every Transformer layer generates a new set of context-aware embeddings.

The representations become richer and more meaningful at every layer.

---

# Summary

The purpose of Self-Attention is **not** to directly predict the next word.

Its purpose is to create **better word representations**.

These new context-aware embeddings are then used by the remaining Transformer layers to perform tasks such as

- Next Word Prediction
- Machine Translation
- Text Summarization
- Question Answering
- Text Classification
- Named Entity Recognition
- Sentiment Analysis

The better the context-aware embeddings, the better the model understands the meaning of the sentence.
