# Member 3 — SVM presentation notes

## Proposed SVM slide

- Binary price-tier classification with labels mapped to `{−1, +1}`.
- From-scratch, single-sample Pegasos minimizes regularized hinge loss.
- Standardized features are required; fixed seeds make training reproducible.
- Optional RFF mapping captures non-linear RBF-like decision boundaries.
- Validation selects hyperparameters; the test split is evaluated once.

Objective:

\[
J(\mathbf{w},b)=
\frac{\lambda}{2}\lVert\mathbf{w}\rVert_2^2+
\frac{1}{n}\sum_{i=1}^{n}
\max\left(0,1-y_i(\mathbf{w}^{\top}\mathbf{x}_i+b)\right)
\]

Learning rate:

\[
\eta_t=\frac{1}{\lambda t}
\]

Suggested visual flow:

```text
standardized X -> optional RFF mapping -> Pegasos -> decision score -> price tier
```

## Speaker notes (approximately 60–90 seconds)

Our second from-scratch model handles the binary price-tier task. During
training, the two original labels are mapped deterministically to minus one
and plus one. Pegasos minimizes the regularized hinge-loss objective using
one randomly selected training sample per iteration. The learning rate
decreases as one over lambda times the iteration number. Every step shrinks
the weight vector, and a margin violation adds a label-directed correction;
the bias is updated but is not regularized.

Because a linear boundary may be too restrictive, we also implement Random
Fourier Features. This approximates an RBF kernel without constructing a full
kernel matrix. The mapping is fitted on the training split once and reused for
validation and test data. We tune lambda and gamma using validation F1, then
evaluate the selected model on the test set exactly once. The synthetic XOR
experiment only verifies that this implementation can learn a non-linear
boundary; it is not a bina.az result.

Real-data placeholders: selected model **[TODO(data-ready)]**, validation F1
**[TODO(data-ready)]**, test F1 **[TODO(data-ready)]**, AUC
**[TODO(data-ready)]**, fit time **[TODO(data-ready)]**, selected $\lambda$
**[TODO(data-ready)]**, selected $\gamma$ **[TODO(data-ready)]**.

## Possible questions

### Why Pegasos?

It optimizes the primal soft-margin objective with simple stochastic updates,
so it is practical to implement transparently with NumPy and does not require
SMO or a quadratic-programming solver.

### Why feature scaling?

Pegasos uses dot products, distances from the decision boundary, and one
shared learning-rate schedule. Standardization prevents large-scale features
from dominating the updates and makes regularization more meaningful.

### What do lambda and gamma do?

Lambda controls regularization strength: larger values penalize large weights
more strongly. Gamma controls the locality of the approximated RBF kernel:
larger values produce more rapidly varying features and more local boundaries.

### Why RFF instead of a full kernel matrix?

RFF creates an explicit finite-dimensional feature representation. We can
then reuse the linear Pegasos solver and avoid storing or optimizing over an
$n\times n$ kernel matrix.

### What is a support vector in this implementation?

We report training samples satisfying
$y_i(\mathbf{w}^{\top}\mathbf{x}_i+b)\leq 1$. These lie on or inside the
margin, including margin-violating samples.

### Why is the synthetic XOR 100% score not the final project result?

XOR is a controlled implementation check with generated data. It shows that
the RFF pipeline can represent a non-linear boundary, but it says nothing
about generalization on the real bina.az dataset.
