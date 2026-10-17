# Member 3 — SVM presentation notes

## Proposed SVM slide

- Binary price-tier classification with labels mapped to `{−1, +1}`.
- NumPy single-sample, Pegasos-style subgradient solver minimizes regularized hinge loss (no projection step).
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

Verified real-data selection: RFF, D=1024, lambda=0.00003, gamma=0.01;
validation F1 **0.845292**, test F1 **0.838028**, test ROC-AUC **0.924372**.
Final objective **0.424093**; weight norm **39.600592**; RFF-space margin
width **0.050504**; **17,528** training samples on/inside the margin.
Full search: 6 linear lambdas
and 16 RFF lambda/gamma pairs, each with 400,000 single-example updates.
There are 45,045 training rows, so this is about 8.88 expected passes, not
400,000 epochs. The update budget and grid both changed since the earlier
linear result: do not attribute the entire improvement to RFF alone.

The teammate's "19x faster" compares RFF Pegasos with **SVC(kernel='linear')**
on their machine, not with an exact RBF SVC, and excludes the complete search.
Our more directly comparable baseline is RBFSampler + SGD hinge, with the same
gamma, D and regularization strength (but a different random map/optimizer).
Do not describe any timing ratio as universal.

## Member 3 bonus: AdaBoost (TODO PDF, page 2, task 20)

The bonus is AdaBoost, not Ridge/SVR (Member 4) or Random Forest (Member 2).
The draft came from the teammate's ZIP; it was reviewed and hardened with
AI assistance. Read the code before accepting responsibility for it.

- A weak learner is a shallow version of the team's NumPy decision tree.
- A boosting round adds one weak learner. This is not an SVM iteration.
- Initially every training row has probability 1/n of being sampled.
- Weighted bootstrap resampling gives hard examples more chances to reappear.
- Error is measured on all original training rows with their current weights,
  not only on the bootstrap sample and never on test rows.
- For weighted error e < 0.5, alpha = 0.5 * log((1-e)/e).
- Update weights by p_i *= exp(-alpha * y_i * h_i), then normalize to sum 1.
  A mistake has y_i*h_i=-1, so its weight increases; a correct answer decreases it.
- Predict using the sign of the weighted vote sum(alpha_t * h_t(x)).
- A score is not a calibrated probability. At exactly zero we select classes_[0].
- Unlucky resamples do not reset the accumulated weights. Perfect learners stop
  early with finite alpha. With no learner better than chance, fit fails clearly.

Full real-data results: depth 3 selected over depth 1; 100 rounds;
validation F1 **0.905720**, test F1 **0.903570**, ROC-AUC **0.970985**.
sklearn discrete AdaBoost reference: test F1 **0.900365**.
Our weighted bootstrap differs from sklearn's exact sample-weighted tree
training; similar scores do not imply mathematically identical fitted models.

Defense answer: "I combined our own trees with discrete adaptive boosting.
Each round increases the sampling weight of mistakes. Depth is selected only
on validation; test labels never choose a depth. sklearn is a separate reference.
The initial draft was teammate/AI-assisted, and I reviewed and tested it."

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
