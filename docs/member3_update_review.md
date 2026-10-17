# Member 3 update review — 2026-10-17

## Scope and provenance

Reviewed the teammate-provided `project_updates.zip` against GitHub main
`7ee9471`. The TODO PDF, page 2, task 20 assigns **AdaBoost to Member 3**;
Random Forest belongs to Member 2, Ridge to Member 4, clustering to Member 1.
The original full assignment statement was not available in the inspected
folders, so this is not a guarantee of a specific bonus mark.

The joint RFF grid and AdaBoost started as teammate/AI-assisted drafts, not
as newly invented implementations in this update. Codex assisted with the
review, fixes, tests, runs and documentation. The member must understand them
before signing. No contribution percentage or personal signature was entered.

## Included changes

- `src/svm_experiments.py`: independent 4-lambda by 4-gamma RFF grid; selected
  RFF lambda is returned rather than the best linear lambda. Full D=1024,
  400,000 updates; fast D=64, 2,000 updates. Record the selected update budget.
- `src/run_all.py`: only the small SVM-budget integration change. Refit/CV
  receive the actual selected budget; fast/full budgets have one source.
- `src/bonus_boosting.py`: NumPy AdaBoost on the team's NumPy tree; weighted
  resampling, weighted error on original rows, log-space normalization,
  finite alpha and early stopping for perfect learners, bounded retry without
  erasing learned sample weights, clear errors for invalid/unfitted inputs.
  Dedicated CLI runs just this bonus and saves JSON and a learning curve.
- `tests/test_svm_joint_grid.py`: all 16 pairs, different linear/RFF best
  lambdas, both possible winning families, test score called once, and budget
  propagation. Existing SVM tests also run.
- `tests/test_bonus_boosting.py`: determinism, learning, stages, string labels,
  input errors, unhelpful learners, perfect learners, preserved retry weights,
  train/validation separation, fast output placement, sklearn import boundary
  and old/new sklearn constructor compatibility.
- `tests/test_run_all.py`: updated the full-budget expectation.
- `tests/test_core_has_no_sklearn.py`: permits the optional module's comparison
  helper; a separate AST test forbids sklearn anywhere else in that module.
- `report/report.tex`, `presentation/member3_svm_notes.md`,
  `contribution_report.md`: update only the Member 3 material/disclosure.

## Real-data verification

Using the existing real CSV and the unchanged main data-preparation code:
64,350 cleaned rows; 171 features; split 45,045 / 9,652 / 9,653;
training median threshold 210,000 AZN; seed 42.

| Model | Validation F1 | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|
| RFF Pegasos, lambda=3e-5, gamma=0.01, D=1024 | 0.845292 | 0.838028 | 0.924372 |
| NumPy AdaBoost, depth 3, 100 rounds | 0.905720 | 0.903570 | 0.970985 |
| sklearn discrete AdaBoost, depth 3, 100 rounds | Not used for selection | 0.900365 | 0.968690 |

Both SVM and AdaBoost were run on the real data in two environments:
the user's Python 3.11 environment (NumPy 1.26.4, pandas 2.2.2, sklearn 1.4.2)
and a separate clean Python 3.14 environment installed from the repository's
exact requirements (NumPy 2.5.4, pandas 3.0.6, sklearn 1.9.1).
Our models' reported validation/test F1 values matched across both environments.
The full test suite passed in both: **231 tests, 40 subtests**. The user's
existing `.venv` was not upgraded or replaced. The bonus-only `--fast` CLI
also completed against the actual CSV in the actual Git checkout.

An additional isolated integration check combined all ZIP code/test changes
with the reviewed Member 3 files and retained the SVM-budget integration:
**242 tests and 40 subtests passed**, one real-data-only test skipped because
that separate snapshot did not contain the CSV. The actual Git checkout's
real-data test passed in both full 231-test runs. A sandbox temporary-folder
permission issue in the first isolated attempt was resolved by using a fresh
writable test directory; it was not a model failure.

Saved local evidence (ignored by Git):
`results/member3-review-20261017/py311/` and
`results/member3-review-20261017/py314/`, each with `svm.json`, `boosting.json`
and the generated figures. These are scoped verification results, not a
replacement for the final shared multi-model CV/timing output.

One important environment difference was reproduced: sklearn RBFSampler+SGD
test F1 was 0.839347 under sklearn 1.4.2 versus 0.833440 under 1.9.1, with
the same seed/data/settings. Use the agreed final environment for report
comparisons. The Python 3.11 linear SVC reference gave F1 0.839115.

The sklearn comparison explicitly uses discrete SAMME on older versions;
on newer versions the removed `algorithm` keyword is omitted. Source:
[official AdaBoostClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html).
Our bootstrap approximation is not the same as sklearn's sample-weighted
split search, so exact equality is not an acceptance criterion.

The full SVM search uses 6 linear candidates plus 16 RFF candidates. All use
400,000 single-example updates, about 8.88 expected training-data passes,
not 400,000 epochs. The old run had only 20,000 updates. Both the grid and the
budget changed, so the F1 gain is not an isolated proof of RFF's contribution.
These are CPU NumPy models; they do not use CUDA or the graphics card.

## Integration handoff for Member 4

Do not blindly overwrite these reviewed files with the original ZIP copies.

1. The ZIP's fast-output separation, benchmark formatting, forest/tree changes,
   other bonus modules, complete rewritten report/slides and shared result
   tables have **not** been mixed into the Member 3 commit. Their owners should
   review and integrate them. Keep Member 3's updated sections and disclosure
   when combining the complete report.
2. Main's original `run_all --fast` still writes shared report outputs, and its
   original `run_bonus()` does not forward fast/data arguments. Apply the ZIP's
   isolation change, also forwarding the selected `--data` path to bonus `run`
   functions that accept `data_path`. Until then use the dedicated AdaBoost
   command below for safe fast bonus runs, not `run_all --fast --bonus`.
3. The ZIP's 0.839 SVC is **linear SVC**, not an exact RBF SVC. The roughly 19x
   timing ratio is a teammate-machine measurement of single-model fit, not
   total grid-search time and not a universal speedup. RBFSampler+SGD is the
   closer RFF comparison, still with different random features/optimizer.
4. Do not claim that sqrt(2/D) alone means RFF necessarily needs smaller lambda:
   it normalizes the map. Independent validation is justified by the changed
   feature geometry. Do not claim the F1 gap is proven to be a threshold issue;
   threshold tuning was not performed.
5. The old main report still has other members' placeholders/old data numbers.
   This scoped change is not a final submission PDF. Regenerate shared report
   tables and PDFs after the owners' integration and final full `run_all`.
6. The repository requirements target Python 3.14, unlike the user's existing
   Python 3.11 `.venv`. Agree a final environment and record it; seed alone is
   not a promise of bit-identical results across package versions/machines.
7. No final full multi-model CV/timing/release run was performed by this scoped
   update. Run it on the final combined tree, then review the report/slides,
   agree contributions and sign personally. Never change existing Git history
   to manufacture dates or contribution evidence.

## Commands from the Git checkout

No activation script or PowerShell execution-policy change is needed:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --disable-warnings
.\.venv\Scripts\python.exe -m src.bonus_boosting --fast
.\.venv\Scripts\python.exe -m src.bonus_boosting
```

The bonus-only CLI writes `results/fast/bonus_boosting.json` and
`results/fast/figures/fig_bonus_boosting.png` for development, or
`results/bonus_boosting.json` and `report/figures/fig_bonus_boosting.png` for full
results. The dataset and generated outputs remain ignored by Git.

The previous uncommitted baseline workflow was preserved in a named Git stash
and its main files copied to ignored `results/member3-pre-update-backup-20261017/`.
The local branch was fast-forwarded to the already-published main; no new
branch commit, push, PR, tag, signature or merge commit was created.

## Commit handoff (not executed)

Review `git diff` and the four new files first. Stage this explicit list, not
`git add .`, to keep other members' later local changes out of your commit:

```powershell
git add -- src/svm_experiments.py src/bonus_boosting.py src/run_all.py tests/test_svm_joint_grid.py tests/test_bonus_boosting.py tests/test_run_all.py tests/test_core_has_no_sklearn.py report/report.tex presentation/member3_svm_notes.md contribution_report.md docs/member3_update_review.md
git diff --cached --stat
git commit -m "feat(svm): fix joint RFF tuning and add reviewed AdaBoost bonus"
git push origin feat/u3-svm
```

This creates no PR. The 28-commit lead visible before your new commit comes
from fast-forwarding to already-published main, not from 28 new agent commits.
If teammates subsequently update shared files, reconcile those changes before
using this handoff; do not force-push or replace their work.
