# Data-generating process (DGP)

_Generator version v2.0, master seed 20261122,_
_2024-01-01 → 2025-12-31._

## Scale

128 businesses across 8 sectors (16 per sector), a 24-month
daily panel, 1,545,074 observed Paybill transactions,
90,446 business-days.

All money is integer **minor units** (1 KSh = 100). The minimum transaction is
**1 KSh** (100 minor units), and the pipeline accepts any positive integer
amount.

## Revenue process (per business, per day)

    lambda_t = base * weekday(dow) * annual(week) * trend(t)
             * exp(shock_t) * latent_quality
    shock_t  = rho * shock_{t-1} + eps_t,   eps_t ~ N(0, sigma_shock)

* `base` — per-business, per-sector level (lognormal around the sector mean).
* `shock_t` — **latent AR(1) shock (rho=0.6, sigma=0.30)** the model cannot
  observe. This is the irreducible error.
* arrivals ~ Poisson(lambda); each arrival spends `lognormal(median_ticket,
  sigma_ticket)` rounded to the nearest shilling, floored at 1 KSh.
* `digital_share` (0.55–0.95, per business) — only this fraction of sales is
  routed through the Paybill and observed. The target is **observed eligible
  Paybill revenue**, not total sales.

## Sectors and gross-margin assumptions (generator assumptions, not measured)

| Sector | median ticket (KSh) | margin bps |
|---|---|---|
| barbershop | 300 | 6000 |
| salon | 500 | 6000 |
| pharmacy | 350 | 2500 |
| general retail / kiosk | 80 | 3500 |
| greengrocer (mama mboga) | 25 | 4000 |
| hardware | 900 | 3000 |
| food vendor | 150 | 4500 |
| second-hand clothing | 600 | 4000 |

The greengrocer and kiosk have a **micro-item share** (single sukuma / onion /
tomato) so their tickets go down to 1 KSh; 8,089 transactions in the panel
are ≤ 1 KSh.

## Non-revenue / disruptive events (injected, then stripped by classification)

refunds, reversals (debit matching a credit), own transfers (payer = business's
own number), capital injections (5–50× median ticket, loan/grant references),
duplicate callbacks, and whole-day Paybill outages.

## Determinism

One master seed; per-business derived seeds (`master + 1,000,003*(i+1)`). The
same seed reproduces byte-identical output (asserted in the test suite).
