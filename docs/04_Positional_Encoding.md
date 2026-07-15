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
# Why Did the Transformer Use Sine and Cosine?

At first glance, it may seem that the only reason for choosing sine and cosine is that their outputs are bounded between **-1 and 1**.

However, this is **not the real reason**.

Many other functions, such as **tanh**, **sigmoid**, and **arctangent**, also produce bounded outputs.

So why did the authors specifically choose **sine** and **cosine**?

Let's understand this step by step.

---

# Requirement 1: Every Position Should Have a Different Representation

Suppose we have the following positions:

```
1
2
3
4
5
```

We need to convert these positions into numerical values.

Many mathematical functions can do this.

For example,

### Square Function

```
f(x) = x²
```

| Position | Output |
|----------|--------|
|1|1|
|2|4|
|3|9|
|4|16|
|5|25|

Every position gets a different value.

Similarly,

### Logarithm

```
f(x) = log(x)
```

| Position | Output |
|----------|--------|
|1|0|
|2|0.69|
|3|1.09|
|4|1.38|

Again,

every position has a different value.

So simply producing different values is **not enough**.

---

# Requirement 2: Values Should Not Become Very Large

Functions like

```
x²
x³
```

grow very quickly.

Example:

```
Position = 1000

x² = 1,000,000
x³ = 1,000,000,000
```

Large values can dominate the word embeddings and make optimization difficult during backpropagation.

Therefore,

the positional values should remain within a reasonable range.

---

# Requirement 3: Why Not Use tanh or Sigmoid?

Let's use

```
tanh(x)
```

| Position | tanh(x) |
|----------|----------|
|1|0.761|
|2|0.964|
|3|0.995|
|4|0.9993|
|5|0.9999|

Initially, everything looks fine.

Each position has a different value.

But now consider larger positions.

| Position | tanh(x) |
|----------|----------|
|10|0.99999999|
|20|≈1|
|50|≈1|
|100|≈1|

Notice what happened.

After a certain point,

every position becomes almost **1**.

This is called **saturation**.

Now,

```
Position 50

Position 500

Position 5000
```

all look almost identical.

The Transformer can no longer distinguish between them.

The same problem occurs with the **sigmoid** function.

---

# Requirement 4: Continuous Variation

Now consider the sine function.

Unlike tanh or sigmoid,

the sine function never becomes constant.

Its values continue changing forever.

Example:

| Position | sin(x) |
|----------|---------|
|1|0.84|
|2|0.91|
|3|0.14|
|4|-0.76|
|5|-0.96|
|6|-0.28|
|7|0.66|

Even for very large positions,

```
Position = 100

Position = 1000

Position = 10000
```

the output continues to change.

The function never saturates.

This allows the Transformer to generate meaningful encodings even for long sequences.

---

# Requirement 5: Preserve Relative Position
There are two concepts:

Absolute Position,
Relative Position,
# Absolute Position:

This tells us the exact location of a word.

Example:

I love cats
Word	Absolute Position
I	1
love	2
cats	3

Here,

"love" is at position 2.
"cats" is at position 3.

This is absolute position.
Relative Position:

Relative position tells us how far apart two words are, not their exact positions.

Example:

I love cats
I      love      cats
1        2         3

Distance between

love and cats = 1

I and cats = 2

I and love = 1

These distances are the relative positions.
That is why relative position is often more important than absolute position.

Many modern Transformers (like T5, DeBERTa, Transformer-XL, etc.) use relative positional encodings because attention often depends more on the distance between words than on their absolute indices.

This is the **most important reason**.

During self-attention,

the Transformer is usually more interested in

> **how far apart two words are**

than their exact positions.

Consider the sentence

```
I love cats
```

Positions

| Word | Position |
|------|----------|
|I|1|
|love|2|
|cats|3|

The distance between

```
love → cats
```

is

```
1
```

Now consider another sentence.

```
Yesterday I love cats
```

Positions become

| Word | Position |
|------|----------|
|Yesterday|1|
|I|2|
|love|3|
|cats|4|

Although the absolute positions changed,

the distance between

```
love → cats
```

is still

```
1
```

The Transformer should understand that

```
love
```

and

```
cats
```

have the same relationship in both sentences.

---

## How Do Sine and Cosine Help?

Sine and cosine satisfy special mathematical identities.

For example,

```
sin(a + b)

=

sin(a)cos(b)

+

cos(a)sin(b)
```

Similarly,

```
cos(a + b)

=

cos(a)cos(b)

−

sin(a)sin(b)
```

These identities allow the positional encoding of

```
Position + Distance
```

to be expressed using the encoding of

```
Position
```

through simple additions and multiplications.

Since self-attention is built using matrix multiplication and linear transformations,

this mathematical property makes it easier for the network to learn **relative positions**.

---

## Why Can't tanh or Sigmoid Do This?

Suppose we use

```
tanh(position)
```

There is no simple identity like

```
tanh(position + distance)

=

Some simple combination of

tanh(position)

and

distance
```

Therefore,

the Transformer would have to learn these relationships entirely from data.

Sine and cosine already contain this mathematical structure.

This makes learning positional relationships easier.

---

# Important Note

The Transformer **does not explicitly calculate**

```
sin(position + distance)
```

during self-attention.

Instead,

the authors designed the positional encoding using sine and cosine because these mathematical properties are naturally embedded in the encoding vectors.

As a result,

the attention mechanism can more easily learn relationships between nearby and distant words.

---

# Final Summary

The Transformer uses sine and cosine because they satisfy several important requirements simultaneously.

✅ Outputs remain bounded.

✅ Values never saturate.

✅ Every position has a unique multi-dimensional representation.

✅ The encoding naturally extends to long sequences.

✅ Nearby positions produce similar patterns.

✅ Mathematical identities make it easier for the model to learn relative positions.

In the next section, we'll learn how the Transformer uses **sine** and **cosine** equations to generate these positional vectors.
