# Market Making Under Information Asymmetry

A Monte Carlo simulation of a market maker facing a mix of informed and uninformed traders. It compares a **Bayesian** market maker, which learns the asset's true value from order flow, against a **naive** market maker that never updates. Both run on identical order flow, so the difference between them measures the value of inference alone.

## Key results

100,000 sessions per scenario, 100 traders per session, spread = 1.0.

| Informed traders | Bayesian P&L | Naive P&L | Bayesian advantage (95% CI) | Cut in informed adverse selection |
|---:|---:|---:|---:|---:|
| 0% | 49.95 | 49.95 | 0.00 | – |
| 5% | 41.27 | 40.55 | 0.72 [0.69, 0.76] | 10.5% |
| 10% | 35.36 | 32.43 | 2.93 [2.88, 2.98] | 23.2% |
| 20% | 26.64 | 19.78 | **6.87** [6.82, 6.91] | **34.0%** |
| 30% | 19.31 | 11.34 | 7.97 [7.94, 8.00] | 33.6% |
| 40% | 13.53 | 5.63 | 7.90 [7.87, 7.93] | 32.3% |
| 50% | 8.48 | 0.81 | 7.67 [7.64, 7.70] | 31.7% |

- **Adverse selection destroys spread income.** With no informed traders, the market maker earns close to the full half-spread on every trade (49.95 per 100 traders). At 50% informed flow, the naive market maker's P&L falls to almost zero (0.81).
- **Bayesian updating recovers much of it.** By inferring the true value from the direction of orders, the Bayesian market maker moves its quotes away from where informed traders profit. At 20% informed flow, this cuts losses to informed traders by 34% and adds 6.87 per session.
- **The value of inference peaks around 30% informed flow.** With few informed traders, there is little to learn and little to lose. With many, both market makers lose heavily and the gap levels off.

![P&L vs information asymmetry](results/pnl_vs_information_asymmetry.png)

![Bayesian advantage](results/bayesian_advantage.png)

![Informed adverse selection](results/informed_adverse_selection.png)

## The model

**Asset.** The true value *V* is either 98 or 102, each with probability 0.5. It is fixed for the session and hidden from the market maker.

**Traders.** 100 traders arrive one at a time. Each is informed with probability α:
- *Informed* traders know *V*. They buy if ask < *V* and sell if bid > *V*; otherwise they don't trade.
- *Uninformed* traders buy or sell with equal probability.

**Quotes.** The market maker quotes symmetrically around an inventory-adjusted reservation price:

```
fair value  = P(V=102)·102 + P(V=98)·98
reservation = fair value − 0.05 × inventory
bid, ask    = reservation ∓ spread / 2
```

Long inventory pushes quotes down to attract buyers; short inventory pushes them up.

**Bayesian update.** After each order, the Bayesian market maker updates its belief with Bayes' theorem. The likelihoods depend on the current quotes, because an informed trader only sends an order when the quote is profitable for them. For a buy:

```
P(buy | V=102) = (1−α)·0.5 + α·1[ask < 102]
P(buy | V=98)  = (1−α)·0.5
P(V=102 | buy) = P(buy | V=102)·P(V=102) / P(buy)
```

Sells are handled symmetrically. The naive market maker keeps P(V=102) = 0.5 throughout.

**P&L.** At the end of each session, the remaining inventory is marked to the true value: `P&L = cash + inventory × V`.

**Adverse selection.** This is the loss on each trade done at a price worse than the true value (selling below *V* or buying above it). The table reports the part caused by informed traders.

## Experimental design

- **Common random numbers.** In each session, both market makers get the same true value and the same random stream. They therefore face the same sequence of trader types, so the P&L difference reflects strategy, not luck.
- **Paired confidence intervals.** The Bayesian advantage is the mean of the per-session P&L differences, with a 95% confidence interval from their standard error.
- **Reproducible.** Every session is seeded, so a rerun gives identical numbers.

## Running it

```bash
pip install -r requirements.txt

python market_maker.py            # full run: 100,000 sessions per scenario (~7 min)
python market_maker.py --quick    # 10,000 sessions per scenario (~40 s)
python market_maker.py --sims 50000 --show   # custom size, display charts
```

Results are saved to `results/`: a CSV with all statistics (mean, standard deviation and 5th percentile of P&L, the advantage and its confidence interval, and adverse selection) and the three charts above.

## Possible extensions

- Let the market maker widen its spread as uncertainty rises, as in Glosten–Milgrom.
- Use a continuous true value, or one that moves during the session.
- Let informed traders choose order size, as in Kyle (1985).
- Optimise the spread and the inventory penalty for each value of α.

---

*Edward Trotman, University of Oxford*
