"""
Market-making under information asymmetry: Bayesian vs naive quoting.

A single market maker quotes a bid and ask on an asset whose true value V is
either 98 or 102 (equal prior). Traders arrive one at a time:

  * informed traders (probability alpha) know V and trade only when the
    quote is mispriced in their favour;
  * uninformed traders buy or sell at random.

Two market makers face identical order flow (common random numbers):

  * Bayesian - updates P(V = 102) after every order using Bayes' theorem,
    accounting for the fact that informed traders only trade on profitable quotes;
  * Naive    - keeps its prior belief and never updates.

Both skew their quotes against inventory. The experiment measures mean P&L,
the Bayesian advantage (with a 95% confidence interval) and adverse-selection
losses across a range of informed-trader shares.

Usage:
    python market_maker.py            # full run (100,000 sessions per alpha, ~7 min)
    python market_maker.py --quick    # 10,000 sessions per alpha (~40 s)
    python market_maker.py --sims 50000 --show
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------

LOW_VALUE, HIGH_VALUE = 98.0, 102.0
VALUES = np.array([LOW_VALUE, HIGH_VALUE])
PRIOR_PROBABILITIES = np.array([0.5, 0.5])
INITIAL_PROB_HIGH = 0.5

NUM_TRADERS = 100
DEFAULT_SIMULATIONS = 100_000
QUICK_SIMULATIONS = 10_000
ALPHA_VALUES = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

SPREAD = 1.0
INVENTORY_PENALTY = 0.05

# Seed offsets: the true value and the order flow use separate, reproducible streams.
VALUE_SEED_OFFSET = 1_000_000
FLOW_SEED_OFFSET = 2_000_000


# ---------------------------------------------------------------------------
# Market-maker components
# ---------------------------------------------------------------------------

def fair_value(prob_high: float) -> float:
    """Expected value of the asset given P(V = HIGH_VALUE)."""
    return prob_high * HIGH_VALUE + (1.0 - prob_high) * LOW_VALUE


def inventory_quotes(fair: float, inventory: int,
                     spread: float = SPREAD,
                     penalty: float = INVENTORY_PENALTY) -> tuple[float, float]:
    """Bid and ask around an inventory-adjusted reservation price.

    Long inventory shifts quotes down (to encourage buyers); short inventory
    shifts them up (to encourage sellers).
    """
    reservation = fair - penalty * inventory
    return reservation - spread / 2.0, reservation + spread / 2.0


def informed_order(true_value: float, bid: float, ask: float) -> str:
    """Informed traders buy below true value and sell above it; otherwise wait."""
    if true_value == HIGH_VALUE and ask < true_value:
        return "buy"
    if true_value == LOW_VALUE and bid > true_value:
        return "sell"
    return "none"


def next_order(true_value: float, bid: float, ask: float,
               alpha: float, rng: np.random.Generator) -> tuple[str, str]:
    """Draw the next trader's type and their order."""
    if rng.random() < alpha:
        return "informed", informed_order(true_value, bid, ask)
    return "uninformed", ("buy" if rng.random() < 0.5 else "sell")


def update_belief(prob_high: float, order: str,
                  bid: float, ask: float, alpha: float) -> float:
    """Posterior P(V = HIGH_VALUE) after observing a buy or sell.

    The likelihoods depend on the current quotes, because an informed trader
    only trades when the quote is profitable for them.
    """
    noise = (1.0 - alpha) * 0.5  # probability an uninformed trader sends this side

    if order == "buy":
        p_given_high = noise + (alpha if ask < HIGH_VALUE else 0.0)
        p_given_low = noise
    elif order == "sell":
        p_given_high = noise
        p_given_low = noise + (alpha if bid > LOW_VALUE else 0.0)
    else:
        raise ValueError("order must be 'buy' or 'sell'")

    evidence = p_given_high * prob_high + p_given_low * (1.0 - prob_high)
    return p_given_high * prob_high / evidence


def adverse_selection(order: str, bid: float, ask: float, true_value: float) -> float:
    """Loss from trading at a price worse than true value (0 if the trade was profitable)."""
    if order == "buy":       # we sold at the ask
        return max(0.0, true_value - ask)
    if order == "sell":      # we bought at the bid
        return max(0.0, bid - true_value)
    return 0.0


# ---------------------------------------------------------------------------
# One trading session
# ---------------------------------------------------------------------------

@dataclass
class SessionResult:
    true_value: float
    final_belief: float
    cash: float
    inventory: int
    pnl: float
    num_trades: int
    informed_trades: int
    adverse_selection: float
    informed_adverse_selection: float


def run_session(true_value: float, alpha: float, rng: np.random.Generator,
                num_traders: int = NUM_TRADERS, use_bayes: bool = True) -> SessionResult:
    """Simulate one market-making session and mark the book to true value at the end."""
    prob_high = INITIAL_PROB_HIGH
    cash, inventory = 0.0, 0
    num_trades = informed_trades = 0
    total_as = informed_as = 0.0

    for _ in range(num_traders):
        bid, ask = inventory_quotes(fair_value(prob_high), inventory)
        trader_type, order = next_order(true_value, bid, ask, alpha, rng)
        if order == "none":
            continue

        loss = adverse_selection(order, bid, ask, true_value)
        total_as += loss
        if trader_type == "informed":
            informed_trades += 1
            informed_as += loss

        if use_bayes:
            prob_high = update_belief(prob_high, order, bid, ask, alpha)

        if order == "buy":       # trader buys from us at the ask
            cash += ask
            inventory -= 1
        else:                    # trader sells to us at the bid
            cash -= bid
            inventory += 1
        num_trades += 1

    return SessionResult(
        true_value=true_value,
        final_belief=prob_high,
        cash=cash,
        inventory=inventory,
        pnl=cash + inventory * true_value,
        num_trades=num_trades,
        informed_trades=informed_trades,
        adverse_selection=total_as,
        informed_adverse_selection=informed_as,
    )


# ---------------------------------------------------------------------------
# Monte Carlo experiment
# ---------------------------------------------------------------------------

def run_experiment(alpha: float, num_sims: int, num_traders: int = NUM_TRADERS) -> dict:
    """Compare Bayesian and naive market makers on matched order flow for one alpha."""
    fields = ("pnl", "adverse_selection", "informed_adverse_selection", "informed_trades")
    bayes = {f: np.empty(num_sims) for f in fields}
    naive = {f: np.empty(num_sims) for f in fields}

    for i in range(num_sims):
        value_rng = np.random.default_rng(VALUE_SEED_OFFSET + i)
        true_value = value_rng.choice(VALUES, p=PRIOR_PROBABILITIES)

        # Same seed for both market makers -> identical trader sequence (common random numbers).
        flow_seed = FLOW_SEED_OFFSET + i
        b = run_session(true_value, alpha, np.random.default_rng(flow_seed), num_traders, use_bayes=True)
        n = run_session(true_value, alpha, np.random.default_rng(flow_seed), num_traders, use_bayes=False)

        for f in fields:
            bayes[f][i] = getattr(b, f)
            naive[f][i] = getattr(n, f)

    diff = bayes["pnl"] - naive["pnl"]
    se = np.std(diff, ddof=1) / np.sqrt(num_sims)

    return {
        "alpha": alpha,
        "bayes_mean_pnl": np.mean(bayes["pnl"]),
        "naive_mean_pnl": np.mean(naive["pnl"]),
        "bayes_std_pnl": np.std(bayes["pnl"], ddof=1),
        "naive_std_pnl": np.std(naive["pnl"], ddof=1),
        "bayes_p5_pnl": np.percentile(bayes["pnl"], 5),
        "naive_p5_pnl": np.percentile(naive["pnl"], 5),
        "pnl_advantage": np.mean(diff),
        "ci_lower": np.mean(diff) - 1.96 * se,
        "ci_upper": np.mean(diff) + 1.96 * se,
        "bayes_adverse_selection": np.mean(bayes["adverse_selection"]),
        "naive_adverse_selection": np.mean(naive["adverse_selection"]),
        "bayes_informed_adverse_selection": np.mean(bayes["informed_adverse_selection"]),
        "naive_informed_adverse_selection": np.mean(naive["informed_adverse_selection"]),
        "bayes_informed_trades": np.mean(bayes["informed_trades"]),
        "naive_informed_trades": np.mean(naive["informed_trades"]),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_csv(results: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def print_table(results: list[dict]) -> None:
    header = (f"{'Alpha':>6} {'Bayes P&L':>10} {'Naive P&L':>10} {'Advantage':>10} "
              f"{'95% CI':>17} {'AS cut':>8}")
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        naive_as = r["naive_informed_adverse_selection"]
        cut = f"{100 * (1 - r['bayes_informed_adverse_selection'] / naive_as):.1f}%" if naive_as else "-"
        ci = f"[{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]"
        print(f"{r['alpha']:>6.2f} {r['bayes_mean_pnl']:>10.2f} {r['naive_mean_pnl']:>10.2f} "
              f"{r['pnl_advantage']:>10.2f} {ci:>17} {cut:>8}")


# Chart styling: two categorical slots from a colour-blind-validated palette.
BAYES_COLOUR = "#2a78d6"
NAIVE_COLOUR = "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")


def _label_line_end(ax, x, y, text, colour) -> None:
    ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=10, color=INK,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colour, lw=1.5))


def save_plots(results: list[dict], out_dir: Path, show: bool) -> list[Path]:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = np.array([r["alpha"] for r in results])
    get = lambda key: np.array([r[key] for r in results])
    line = dict(linewidth=2, marker="o", markersize=6, markeredgecolor="white", markeredgewidth=1.5)
    paths = []

    # 1. Mean P&L
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(a, get("bayes_mean_pnl"), color=BAYES_COLOUR, label="Bayesian", **line)
    ax.plot(a, get("naive_mean_pnl"), color=NAIVE_COLOUR, label="Naive", **line)
    _label_line_end(ax, a[-1], get("bayes_mean_pnl")[-1], "Bayesian", BAYES_COLOUR)
    _label_line_end(ax, a[-1], get("naive_mean_pnl")[-1], "Naive", NAIVE_COLOUR)
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(right=a[-1] + 0.07)
    _style_axes(ax, "Share of informed traders", "Mean P&L per 100-trader session",
                "Market-maker P&L falls as informed trading rises")
    fig.tight_layout()
    paths.append(out_dir / "pnl_vs_information_asymmetry.png")
    fig.savefig(paths[-1], dpi=200)

    # 2. Bayesian advantage with 95% CI
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(a, get("ci_lower"), get("ci_upper"), color=BAYES_COLOUR, alpha=0.2, linewidth=0)
    ax.plot(a, get("pnl_advantage"), color=BAYES_COLOUR, **line)
    ax.axhline(0, color=MUTED, linewidth=1, linestyle="--")
    _style_axes(ax, "Share of informed traders", "Bayesian minus naive P&L (95% CI)",
                "Value of Bayesian inference")
    fig.tight_layout()
    paths.append(out_dir / "bayesian_advantage.png")
    fig.savefig(paths[-1], dpi=200)

    # 3. Informed-trader adverse selection
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(a, get("bayes_informed_adverse_selection"), color=BAYES_COLOUR, label="Bayesian", **line)
    ax.plot(a, get("naive_informed_adverse_selection"), color=NAIVE_COLOUR, label="Naive", **line)
    _label_line_end(ax, a[-1], get("bayes_informed_adverse_selection")[-1], "Bayesian", BAYES_COLOUR)
    _label_line_end(ax, a[-1], get("naive_informed_adverse_selection")[-1], "Naive", NAIVE_COLOUR)
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(right=a[-1] + 0.07)
    _style_axes(ax, "Share of informed traders", "Mean loss to informed traders per session",
                "Adverse selection from informed traders")
    fig.tight_layout()
    paths.append(out_dir / "informed_adverse_selection.png")
    fig.savefig(paths[-1], dpi=200)

    if show:
        plt.show()
    plt.close("all")
    return paths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sims", type=int, default=DEFAULT_SIMULATIONS,
                   help=f"sessions per alpha (default {DEFAULT_SIMULATIONS:,})")
    p.add_argument("--quick", action="store_true",
                   help=f"shortcut for --sims {QUICK_SIMULATIONS:,}")
    p.add_argument("--traders", type=int, default=NUM_TRADERS,
                   help=f"traders per session (default {NUM_TRADERS})")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results",
                   help="output folder for the CSV and charts")
    p.add_argument("--show", action="store_true", help="display charts on screen")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    num_sims = QUICK_SIMULATIONS if args.quick else args.sims
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Monte Carlo: {num_sims:,} sessions x {args.traders} traders per alpha")
    results = []
    for alpha in ALPHA_VALUES:
        print(f"  running alpha = {alpha:.2f} ...", flush=True)
        results.append(run_experiment(alpha, num_sims, args.traders))

    print_table(results)
    csv_path = args.out / "market_maker_results.csv"
    save_csv(results, csv_path)
    plots = save_plots(results, args.out, args.show)

    print("\nSaved:")
    for path in (csv_path, *plots):
        print(f"  {path}")


if __name__ == "__main__":
    main()
