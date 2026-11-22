# Repayment simulation

_A policy simulation on synthetic data — not a lending result._

For each business-month in the test partitions, the engine assesses the business
using only data before the cut-off, then simulates collection from the actual
future eligible receipts (`C_t = min(E_t × recovery, balance, daily cap)`, fee
first, round down).

## Outcomes

| Result | Value |
|---|---|
| assessments | 1652 |
| eligible | 1651 |
| settled within term | 1215 |
| extended | 423 |
| fallback | 13 |
| median days to settle | 25.0 |
| total advanced | KSh 25,903,800 |
| total shortfall | KSh 34,951 |
| loss rate (simulated) | 0.13% |

## Binding constraint

{
  "affordability": 1258,
  "policy_max": 364,
  "rate": 29
}

`affordability` binds ~75% of advances, `policy_max` ~23%, `rate` ~2%. The model
is doing real work — not 90% `policy_max`.

## Stress test (future revenue −30%)

Outcomes: {"extended": 906, "settled": 602, "fallback": 143}; simulated loss rate
1.50% (vs 0.13% normal).

The engine **does not** "reduce defaults" — there are no defaults, only simulated
shortfalls on synthetic data.
