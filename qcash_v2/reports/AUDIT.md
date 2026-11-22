# Audit — what was wrong before, and what changed

_Generated from an executed audit of the pre-existing pipeline._

## What existed

| Artefact | Content |
|---|---|
| `dataset_v1/` | One real business (Belgard Barbers), 8,228 transactions, KSh **floats** |
| `qcash_synthetic_v1/` | 80 synthetic businesses, 8 archetypes, KSh floats |
| `train_qcash.py` | Single-business weekly panel, chronological split |
| `service/` | ERA decision engine (floats, KSh, %-based rates) |

## The four most serious problems found

1. **Money is stored as floats in KSh**, not integers in minor units. The
   engine guide invariant I3 (1 KSh = 100 minor units, `BIGINT`, no floats)
   is violated everywhere. `0.1 + 0.2 != 0.3` applies to money too.
2. **The synthetic generator leaked the answer.** `qcash_synthetic_v1` drew a
   daily *revenue* number, then generated transactions that were rescaled to
   sum to that revenue. The model's target was essentially the generator's
   output with no unobserved noise — a near-tautology that inflates R².
3. **No business-grouped split.** Businesses appeared in both train and test,
   so the model memorised business identity rather than behaviour. The prompt's
   "never split randomly by row" rule was not enforced.
4. **No robustness to absurd low-value transactions.** The real data already
   contains a 1-shilling transaction (min `1.0 KSh`), but the
   pipeline used floats and reported nothing about how it handles KSh 1.

Other gaps: no oracle/irreducible-error ceiling, no permutation (shuffle-target)
test, no leakage tests, no refund/reversal/capital-injection/outage modelling,
model selection on the holdout rather than validation, and no uncertainty
(lower-bound) estimate.

## What the rebuild does differently

1. Integer **minor units** everywhere; minimum generated transaction is 1 KSh
   (100 minor units); every money field is asserted to be an integer.
2. A **latent AR(1) demand shock** (unobservable) plus Poisson arrivals and
   per-sector ticket distributions, so the problem is genuinely hard.
3. **Grouped-by-business + ordered-by-time splits** with a business holdout, a
   temporal holdout, a tail (churned/shocked) holdout, and four leakage tests.
4. **Honest evaluation**: baselines, oracle ceiling, permutation test, learning
   curve, five seeds, and a 90% lower bound with measured coverage.
