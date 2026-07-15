# Multi-Head Attention - Why Do We Need Multiple Heads?

So far, we have learned

- Word Embeddings
- Positional Encoding
- Self-Attention

Now let's ask an important question.

> If Self-Attention already works, why did the Transformer introduce **Multi-Head Attention**?

Why not simply use one Self-Attention mechanism?

To answer this, let's first understand the limitation of using a single attention head.

---

# How Does a Single Attention Head Work?

Suppose we have the sentence

```
The animal didn't cross the street because it was tired.
```

When processing the word

```
it
```

Self-attention tries to determine

> Which previous word does "it" refer to?

Ideally,

```
it

↓

animal
```

The attention mechanism learns this relationship.

This is good.

---

# But Is That the Only Relationship?

Not at all.

The same sentence contains many different relationships.

```
animal  ----> it
```

Pronoun resolution.

```
cross ----> street
```

Action and location.

```
didn't ----> cross
```

Negation.

```
because ----> tired
```

Cause and effect.

All these relationships exist **at the same time**.

---

# The Problem with a Single Attention Head

A single attention head produces **only one attention matrix**.

That means,

for every word,

it learns **one pattern of attention**.

Although this attention may capture an important relationship,

it cannot specialize in every possible relationship simultaneously.

Think of it like one person trying to observe everything happening in a busy city.

They can focus on one thing,

but they will miss many others.

---

# Human Analogy

Imagine a detective investigating a crime.

One detective tries to

- interview witnesses
- examine fingerprints
- analyze CCTV footage
- collect DNA evidence
- reconstruct the timeline

all by themselves.

This is possible,

but extremely difficult.

Now imagine instead,

we assign specialists.

Detective 1

→ Witness interviews

Detective 2

→ Fingerprints

Detective 3

→ CCTV analysis

Detective 4

→ DNA evidence

Detective 5

→ Timeline reconstruction

Each detective becomes an expert in one aspect of the investigation.

Finally,

all their findings are combined.

The overall investigation becomes much stronger.

Multi-Head Attention follows exactly the same idea.

---

# Multiple Attention Heads

Instead of creating

```
One Attention Head
```

the Transformer creates several attention heads.

Example

```
Head 1

Head 2

Head 3

Head 4

...

Head h
```

Each head receives the **same input sentence**.

However,

each head learns **different relationships** during training.

No one explicitly programs these behaviors.

They emerge automatically while training the model.

---

# What Might Different Heads Learn?

Although there are no fixed rules,

research has shown that different heads often specialize in different tasks.

For example,

### Head 1

May learn

```
Subject → Verb
```

Example

```
Dogs

↓

bark
```

---

### Head 2

May learn

```
Pronoun Resolution
```

Example

```
animal

↓

it
```

---

### Head 3

May learn

```
Long-distance relationships
```

Example

```
book

↓

interesting
```

even if many words appear between them.

---

### Head 4

May learn

```
Modifier relationships
```

Example

```
beautiful

↓

flower
```

---

### Head 5

May learn

```
Verb → Object
```

Example

```
eat

↓

apple
```

Each head observes the sentence from a different perspective.

Together,

they capture much richer information than a single attention head.

---

# An Important Point

The Transformer **does not assign these roles manually.**

We never tell

```
Head 1

↓

Learn grammar.
```

or

```
Head 2

↓

Learn pronouns.
```

Instead,

during backpropagation,

each head gradually learns whatever relationships help minimize the training loss.

These specializations emerge automatically.

---

# Visual Intuition

Instead of

```
Sentence

↓

Self-Attention

↓

Output
```

the Transformer performs

```
Sentence

↓

Head 1

↓

Output 1


Sentence

↓

Head 2

↓

Output 2


Sentence

↓

Head 3

↓

Output 3


...

Sentence

↓

Head h

↓

Output h
```

All outputs are then combined.

---

# How Are They Combined?

The outputs of every attention head are

```
Concatenated
```

into one large vector.

```
Head1 Output

+

Head2 Output

+

Head3 Output

+

...

↓

Concatenation
```

A final linear layer then mixes the information from all heads.

This becomes the final output of the Multi-Head Attention layer.

---

# Why Not Simply Increase the Size of One Attention Head?

This is another common question.

Suppose instead of using

```
8 heads × 64 dimensions
```

we create

```
1 head × 512 dimensions
```

Wouldn't both contain the same number of parameters?

Not exactly.

One large attention head still learns **one attention pattern**.

Multiple smaller heads can learn **multiple independent attention patterns simultaneously**.

This diversity is the key advantage of Multi-Head Attention.

---

# Summary

Self-Attention allows every word to attend to every other word.

However,

a single attention head can learn only one attention pattern at a time.

Different linguistic relationships exist simultaneously within a sentence.

By creating multiple attention heads,

the Transformer allows each head to specialize in different relationships.

The outputs of all heads are then combined,

giving the model a much richer understanding of the sentence.

This is why the Transformer uses **Multi-Head Attention** instead of a single attention mechanism.

---

# What's Next?

Now that we understand **why** Multi-Head Attention is needed,

the next chapter explains **how it is implemented mathematically.**

We'll answer questions such as

- Why do we create separate Q, K, and V matrices for each head?
- Why is `d_model` divided by the number of heads?
- Why are the outputs concatenated?
- Why is another linear layer applied after concatenation?

These questions will become much easier now that we understand the motivation behind Multi-Head Attention.
