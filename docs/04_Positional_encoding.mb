# Positional Encoding in Transformers

## Why Do We Need Positional Encoding?

Before Transformers, sequence models like **RNNs**, **LSTMs**, and **GRUs** processed one word at a time.

Example:

```
I → love → cats
```

Since words are processed sequentially, these models naturally understand the order of words.

---

### But Transformers Work Differently

One of the biggest advantages of Transformers is **parallel processing**.

Instead of processing words one after another, the Transformer processes **all words simultaneously**.

Example:

```
Sentence:

I love cats
```

Instead of

```
I
↓
love
↓
cats
```

the Transformer receives

```
I     love     cats
│       │        │
└───────┼────────┘
     Processed Together
```

This parallel processing makes Transformers much faster than RNNs.

---

## The Problem

Since every word is processed at the same time, the Transformer does **not know the order of words**.

For example,

```
I love cats
```

and

```
Cats love I
```

contain the same words.

Without positional information, the Transformer only sees

```
I
love
cats
```

It has no idea which word comes first, second, or third.

However, word order is extremely important because it changes the meaning of a sentence.

For example,

```
Dog bites man
```

is completely different from

```
Man bites dog
```

Therefore, we need a way to preserve the position of every word.

This is exactly why **Positional Encoding** is introduced.

---

# Simple Approaches to Preserve Position

Before understanding the Transformer's solution, let's see two simple approaches and why they fail.

---

# Method 1: Assign Position Numbers (Unbounded Values)

Simply assign each word its position.

Example:

```
Sentence:

I love cats
```

| Word | Position |
|------|----------|
| I | 1 |
| love | 2 |
| cats | 3 |

Another sentence:

```
Today I love beautiful little white cats
```

| Word | Position |
|------|----------|
| Today | 1 |
| I | 2 |
| love | 3 |
| beautiful | 4 |
| little | 5 |
| white | 6 |
| cats | 7 |

For very long documents,

```
Word 1
Word 2
...
Word 1000
...
Word 50000
```

The position numbers become

```
1,2,3,4,...50000
```

These values keep increasing without any upper limit.

This is called **Unbounded Values**.

---

## Why is this a problem?

During training, neural networks perform many matrix multiplications and gradient updates during backpropagation.

Word embeddings usually contain small values like

```
0.12
-0.45
0.87
```

But position numbers may become

```
1
500
5000
20000
```

The large position values dominate the embedding values.

As a result,

- Training becomes unstable.
- Gradients may become difficult to optimize.
- Learning becomes inefficient.

So directly using position numbers is **not a good solution**.

---

# Method 2: Normalize the Position

To avoid large numbers, normalize the position.

Formula:

```
Normalized Position = Position / Sentence Length
```

Example 1

Sentence:

```
Thank you
```

Length = 2

| Word | Position | Normalized Position |
|------|----------|---------------------|
| Thank | 1 | 1/2 = 0.5 |
| you | 2 | 2/2 = 1 |

---

Example 2

Sentence:

```
How are you
```

Length = 3

| Word | Position | Normalized Position |
|------|----------|---------------------|
| How | 1 | 1/3 = 0.33 |
| are | 2 | 2/3 = 0.66 |
| you | 3 | 3/3 = 1 |

Now every value lies between

```
0 and 1
```

This removes the problem of unbounded values.

---

## But Another Problem Appears

The normalized value now depends on the **sentence length**.

Consider the **second word**.

Sentence A

```
Length = 2
```

Second word

```
2/2 = 1
```

Sentence B

```
Length = 3
```

Second word

```
2/3 = 0.66
```

Sentence C

```
Length = 5
```

Second word

```
2/5 = 0.40
```

The **same position** now has different values in different sentences.

Likewise,

The first word becomes

```
Length 2 → 0.5
Length 5 → 0.2
Length 10 → 0.1
```

So the position representation is **not consistent** across different sentence lengths.

Therefore, normalization is also **not a reliable solution**.

---

# Summary of Both Methods

## Method 1

Assign position directly.

```
1,2,3,4,...
```

### Advantages

- Very simple.

### Disadvantages

- Unbounded values.
- Large numbers affect backpropagation.
- Training becomes difficult.

---

## Method 2

Normalize positions.

```
Position / Sentence Length
```

### Advantages

- Values remain between 0 and 1.
- No large numbers.

### Disadvantages

- Position depends on sentence length.
- Same position gets different values in different sentences.
- Position representation is inconsistent.

---

# Transformer's Solution

Instead of using

```
1
2
3
```

or

```
0.33
0.66
1
```

the Transformer converts every position into a unique vector using **sine and cosine functions**.

These functions

- produce values between **-1 and 1**
- provide unique patterns for every position
- preserve relative distances between words
- work for both short and long sentences
- produce consistent positional representations

This technique is called **Sinusoidal Positional Encoding**.

In the next section, we'll learn how the Transformer uses **sine** and **cosine** equations to generate these positional vectors.
