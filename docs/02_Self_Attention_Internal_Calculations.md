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

Performing similar calculations for "love" and "cats":

Q = \[ 0.04 0.12 0.08 0.12 \\ 0.09 0.21 0.13 0.17 \\ 0.12 0.02 0.20
0.13\]

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

Performing similar calculations for "love" and "cats":

K = \[ 0.08 0.04 0.15 0.06 \\ 0.18 0.13 0.25 0.14 \\ 0.20 0.11 0.19
0.11\]

-   **Dimensions of K:** `(3, 4)`

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

Performing similar calculations for "love" and "cats":

V = \[ 0.10 0.04 0.08 0.08 \\ 0.21 0.12 0.19 0.20 \\ 0.08 0.10 0.11
0.02\]

-   **Dimensions of V:** `(3, 4)`

------------------------------------------------------------------------

### **4. Attention Score: Q × Kᵀ**

This is where the magic of "attention" happens. For each word (as a
Query), we calculate how much it relates to every other word (as a Key).

**Why dot product (Q × Kᵀ) is used:**

-   **What problem it solves:** We need a way to measure the
    *similarity* or *relatedness* between a Query vector and a Key
    vector.
-   **Intuition behind similarity:** The dot product of two vectors is
    high when they point in similar directions and have large
    magnitudes. If vectors are normalized (unit length), the dot product
    becomes the cosine of the angle between them, which is a common
    measure of similarity. In Self-Attention, the learned
    transformations WQ and WK are designed such that the dot product in
    their respective spaces reveals meaningful semantic relationships.
-   **What happens without it:** Without a similarity measure, we
    wouldn't know which words are relevant to each other, and the core
    purpose of attention (selectively focusing on relevant information)
    would be lost.

**The calculation:** We multiply the Query matrix (Q) by the transpose
of the Key matrix (Kᵀ).

-   `Q`: `(seq_len, d_k)` = `(3, 4)`
-   `K^T`: `(d_k, seq_len)` = `(4, 3)` (transpose of K which is
    `(3, 4)`)
-   `Attention_Scores`: `(seq_len, seq_len)` = `(3, 3)`

The result will be a `(3, 3)` matrix, where each element `(i, j)`
represents the attention score of Query `i` looking at Key `j`.

`\text{Attention_Scores}`{=tex} = Q × Kᵀ

`\text{Attention_Scores}`{=tex} = \[ 0.04 0.12 0.08 0.12 \\ 0.09 0.21
0.13 0.17 \\ 0.12 0.02 0.20 0.13\] × \[ 0.08 0.18 0.20 \\ 0.04 0.13 0.11
\\ 0.15 0.25 0.19 \\ 0.06 0.14 0.11\]

Let's calculate the first row (attention scores for "I" looking at "I",
"love", "cats"):

-   **"I" (Query 1) vs "I" (Key 1):** `[0.04, 0.12, 0.08, 0.12]` ·
    `[0.08, 0.04, 0.15, 0.06]` = (0.04 × 0.08) + (0.12 × 0.04) + (0.08 ×
    0.15) + (0.12 × 0.06) = 0.0032 + 0.0048 + 0.0120 + 0.0072 = 0.0272

-   **"I" (Query 1) vs "love" (Key 2):** `[0.04, 0.12, 0.08, 0.12]` ·
    `[0.18, 0.13, 0.25, 0.14]` = (0.04 × 0.18) + (0.12 × 0.13) + (0.08 ×
    0.25) + (0.12 × 0.14) = 0.0072 + 0.0156 + 0.0200 + 0.0168 = 0.0596

-   **"I" (Query 1) vs "cats" (Key 3):** `[0.04, 0.12, 0.08, 0.12]` ·
    `[0.20, 0.11, 0.19, 0.11]` = (0.04 × 0.20) + (0.12 × 0.11) + (0.08 ×
    0.19) + (0.12 × 0.11) = 0.0080 + 0.0132 + 0.0152 + 0.0132 = 0.0496

So, the first row of Attention Scores is `[0.0272, 0.0596, 0.0496]`

Performing similar calculations for the other rows:

`\text{Attention_Scores}`{=tex} = \[ 0.0272 0.0596 0.0496 \\ 0.0594
0.1287 0.1083 \\ 0.0468 0.0964 0.0886\]

-   **Dimensions of Attention_Scores:** `(3, 3)`

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

Let's calculate √dk: `d_k = 4` `sqrt(d_k) = sqrt(4) = 2`

Now, divide every element in our `Attention_Scores` matrix by 2:

`\text{Scaled_Attention_Scores}`{=tex} = `\frac{1}{√dk}`{=tex} ×
`\text{Attention_Scores}`{=tex}

`\text{Scaled_Attention_Scores}`{=tex} = `\frac{1}{2}`{=tex} × \[ 0.0272
0.0596 0.0496 \\ 0.0594 0.1287 0.1083 \\ 0.0468 0.0964 0.0886\]

`\text{Scaled_Attention_Scores}`{=tex} = \[ 0.0136 0.0298 0.0248 \\
0.0297 0.06435 0.05415 \\ 0.0234 0.0482 0.0443\]

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
-   **How it works:** Softmax takes an array of numbers and squashes
    them into a probability distribution where all values are between 0
    and 1, and their sum equals 1. The formula is:
    `\text{Softmax}`{=tex}(z_i) =
    `\frac{e^{z_i}}{\sum_{j} e^{z_j}}`{=tex}

We apply Softmax row-wise. Each row in our `Scaled_Attention_Scores`
matrix represents how one word (Query) attends to all other words
(Keys).

Let's calculate the Softmax for the first row (attention weights for
"I"): `[0.0136, 0.0298, 0.0248]`

1.  **Calculate e\^{z_i} for each element:**

    -   e\^{0.0136} `\approx 1.01369`{=tex}
    -   e\^{0.0298} `\approx 1.03025`{=tex}
    -   e\^{0.0248} `\approx 1.02511`{=tex}

2.  **Calculate the sum of exponentials (`\sum`{=tex}e\^{z_j}):**
    1.01369 + 1.03025 + 1.02511 = 3.06905

3.  **Divide each e\^{z_i} by the sum:**

    -   0.0136 → 1.01369 / 3.06905 `\approx 0.3303`{=tex}
    -   0.0298 → 1.03025 / 3.06905 `\approx 0.3357`{=tex}
    -   0.0248 → 1.02511 / 3.06905 `\approx 0.3340`{=tex}

So, the attention weights for "I" are `[0.3303, 0.3357, 0.3340]`. Notice
these sum to approximately 1.0.

Performing similar calculations for the other rows:

`\text{Attention_Weights}`{=tex} =
`\text{Softmax}`{=tex}(`\text{Scaled_Attention_Scores}`{=tex})

`\text{Attention_Weights}`{=tex} = \[ 0.3303 0.3357 0.3340 \\ 0.3292
0.3421 0.3287 \\ 0.3306 0.3384 0.3310\]

-   **Dimensions of Attention_Weights:** `(3, 3)`
    -   Each row `i` tells us how much word `i` ("I", "love", or "cats")
        pays attention to word 1 ("I"), word 2 ("love"), and word 3
        ("cats").

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
