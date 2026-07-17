## Introduction

In the previous chapter, we learned how the **Residual (Skip) Connection** combines the original input with the output of the Multi-Head Attention layer.

The result of this addition is then passed to **Layer Normalization**.

In this chapter, we will understand how Layer Normalization works internally by performing every calculation step by step.

---

# Why Do We Need Gamma (γ) and Beta (β)?

At first glance, Layer Normalization seems complete after normalization.

```text
Input

↓

Normalize

↓

Output
```

After normalization,

- Mean becomes approximately **0**
- Standard Deviation becomes approximately **1**

For example,

```text
Before Normalization

[7.0, 3.2, 5.1, 5.0]
```

After Normalization

```text
[1.43, -1.40, 0.02, -0.06]
```

The values are now centered around zero and have a consistent scale.

So a natural question is:

> **If the values are already normalized, why do we need Gamma (γ) and Beta (β)?**

---

# What Happens Without Gamma and Beta?

Imagine every Layer Normalization layer always produced outputs like this.

```text
[-1.2, 0.4, 1.5, -0.7]

↓

Always Mean = 0

Always Standard Deviation = 1
```

Every Transformer layer would be forced to output features with exactly the same distribution.

The model would **lose the freedom to decide**:

- Which features should be stronger?
- Which features should be weaker?
- Should a feature be shifted slightly upward or downward?

Normalization would force every feature into the same range.

This can make the model less expressive.

---

# Why Researchers Added Gamma (γ)

Researchers wanted the stability provided by normalization **without restricting what the model could learn**.

So they introduced a learnable scaling parameter called **Gamma (γ)**.

Gamma allows the model to decide

- Increase the importance of this feature.
- Decrease the importance of this feature.
- Leave this feature unchanged.

For example,

Suppose the normalized output is

```text
[1.43, -1.40, 0.02, -0.06]
```

and

```text
γ

=

[2.0, 0.5, 1.0, 1.5]
```

After Gamma scaling,

```text
[2.86, -0.70, 0.02, -0.09]
```

Notice that

- the first feature became more important,
- the second feature became less important,
- the third feature stayed the same.

The model learns these scaling values automatically during training.

---

# Why Researchers Added Beta (β)

Scaling alone is not always enough.

Sometimes the model learns better if the features are shifted upward or downward.

Beta provides this flexibility.

Suppose

```text
Gamma Output

[2.86, -0.70, 0.02, -0.09]
```

and

```text
β

=

[0.2, -0.3, 0.5, 0.1]
```

After adding Beta,

```text
[3.06, -1.00, 0.52, 0.01]
```

Beta shifts each feature without changing the relationships learned during normalization.

---

# Why Not Skip Normalization Completely?

Without Layer Normalization,

- activations can become too large or too small,
- training becomes unstable,
- gradients may explode or vanish,
- deep Transformers become much harder to train.

Normalization keeps training stable.

Gamma and Beta restore the flexibility that normalization removes.

---

# The Main Idea

Think of Layer Normalization as two separate steps.

**Step 1**

Normalize the values so training becomes stable.

```text
Input

↓

Normalize
```

**Step 2**

Allow the model to adjust those normalized values.

```text
Normalize

↓

Gamma (Scale)

↓

Beta (Shift)
```

Normalization provides **stability**.

Gamma and Beta provide **flexibility**.

Together, they allow Transformers to train efficiently while still learning powerful feature representations.

# Input to Layer Normalization

From the previous chapter, the output of the Residual Connection is

```text
Original Input (Embedding + Positional Encoding)

[5.2, 0.7, 2.1, 3.4]

+

Context-Aware Output

[1.8, 2.5, 3.0, 1.6]

=

Residual Output

[7.0, 3.2, 5.1, 5.0]
```

This Residual Output becomes the input to Layer Normalization.

```text
LayerNorm Input

[7.0, 3.2, 5.1, 5.0]
```

Shape

```text
1 × 4
```

---

# Step 1: Calculate the Mean

The mean is the average of all values.

```text
Mean

=

(7.0 + 3.2 + 5.1 + 5.0)

/

4

=

20.3 / 4

=

5.075
```

---

# Step 2: Calculate the Variance

First, calculate the squared difference between each value and the mean.

| Value | Value − Mean | Square |
|------:|-------------:|--------:|
|7.0|1.925|3.7056|
|3.2|-1.875|3.5156|
|5.1|0.025|0.0006|
|5.0|-0.075|0.0056|

Now calculate the variance.

```text
Variance

=

(3.7056 + 3.5156 + 0.0006 + 0.0056)

/

4

=

7.2274 / 4

=

1.8069
```

---

# Step 3: Calculate the Standard Deviation

The standard deviation is the square root of the variance.

```text
Standard Deviation

=

√1.8069

≈

1.344
```

---

# Step 4: Normalize the Values

The normalization formula is

```text
Normalized Value

=

(Value − Mean)

/

Standard Deviation
```

### First Value

```text
(7.0 − 5.075) / 1.344

=

1.43
```

### Second Value

```text
(3.2 − 5.075) / 1.344

=

-1.40
```

### Third Value

```text
(5.1 − 5.075) / 1.344

=

0.02
```

### Fourth Value

```text
(5.0 − 5.075) / 1.344

=

-0.06
```

The normalized output becomes

```text
[1.43, -1.40, 0.02, -0.06]
```

---

# Step 5: Gamma (γ) Scaling

Suppose the model has learned the following Gamma values.

```text
γ

=

[1.1, 0.9, 1.2, 0.8]
```

Multiply each normalized value by its corresponding Gamma.

```text
1.43 × 1.1 = 1.573

-1.40 × 0.9 = -1.260

0.02 × 1.2 = 0.024

-0.06 × 0.8 = -0.048
```

After Gamma Scaling

```text
[1.573, -1.260, 0.024, -0.048]
```

---

# Why Do We Need Gamma (γ)?

After normalization, every feature has approximately the same scale.

However, not every feature contributes equally to the prediction.

Gamma allows the Transformer to learn **how much importance should be given to each feature**.

For example,

- A larger Gamma increases a feature's influence.
- A smaller Gamma reduces a feature's influence.

Instead of forcing every feature to have the same scale, the model learns the most useful scale during training.

---

# Step 6: Beta (β) Shift

Suppose the model has learned

```text
β

=

[0.2, -0.1, 0.3, 0.1]
```

Add Beta to each value.

```text
1.573 + 0.2 = 1.773

-1.260 + (-0.1) = -1.360

0.024 + 0.3 = 0.324

-0.048 + 0.1 = 0.052
```

Final LayerNorm Output

```text
[1.773, -1.360, 0.324, 0.052]
```

---

# Why Do We Need Beta (β)?

Normalization centers the values around zero.

However, the best representation for a model may not always be centered exactly at zero.

Beta allows the model to **shift each feature upward or downward** by a learned amount.

This gives the Transformer additional flexibility to learn better feature representations.

---

# Complete Layer Normalization Pipeline

```text
Residual Output

[7.0, 3.2, 5.1, 5.0]

↓

Calculate Mean

↓

Calculate Variance

↓

Calculate Standard Deviation

↓

Normalize

↓

Multiply by Gamma (γ)

↓

Add Beta (β)

↓

LayerNorm Output

[1.773, -1.360, 0.324, 0.052]
```

---

# Key Takeaways

- Layer Normalization first normalizes the input values.
- Gamma (γ) learns the best scale for each feature.
- Beta (β) learns the best shift for each feature.
- Gamma and Beta are trainable parameters learned during backpropagation.
- The LayerNorm output is passed to the Feed Forward Network (FFN).
