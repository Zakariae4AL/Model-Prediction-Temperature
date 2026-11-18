"""Regenerate the README figures: python scripts/make_figures.py

Uses reports/scores.csv (from `tempforecast evaluate --output reports/scores.csv`)
and the bundled data. Every number in the figures is computed, nothing is typed in.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tempforecast import data, evaluate  # noqa: E402
from tempforecast.models import ConformalBand, EnsembleForecaster  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
HOT, COLD, GREY, INK = "#e8590c", "#1c7ed6", "#868e96", "#212529"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.edgecolor": "#adb5bd",
        "text.color": INK,
        "axes.labelcolor": INK,
    }
)


def v1_vs_v2(scores: pd.DataFrame) -> None:
    day1 = scores[scores["horizon"] == 1].pivot(index="model", columns="target", values="mae")
    order = [
        "v1 · weekday regression",
        "Climatology",
        "Persistence",
        "Ridge regression",
        "Gradient boosting",
        "Ensemble (ridge + boosting)",
    ]
    day1 = day1.loc[order]
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(day1))
    ax.barh(y + 0.2, day1["tmax"], 0.38, color=HOT, label="Max temperature")
    ax.barh(y - 0.2, day1["tmin"], 0.38, color=COLD, label="Min temperature")
    for i, (mx, mn) in enumerate(zip(day1["tmax"], day1["tmin"], strict=True)):
        ax.text(mx + 0.08, i + 0.2, f"{mx:.2f}", va="center", fontsize=8.5)
        ax.text(mn + 0.08, i - 0.2, f"{mn:.2f}", va="center", fontsize=8.5)
    ax.set_yticks(y, day1.index)
    ax.set_xlabel("Mean absolute error on tomorrow's temperature (°C) — lower is better")
    ax.set_title("Tomorrow's forecast error, test years 2023–2026")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(0, day1.to_numpy().max() * 1.12)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "v1_vs_v2.png")


def by_horizon(scores: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    styles = {
        "Persistence": dict(color=GREY, ls="--"),
        "Climatology": dict(color=GREY, ls=":"),
        "Ensemble (ridge + boosting)": dict(color=INK, lw=2.2),
    }
    for ax, (target, label) in zip(axes, [("tmax", "Max temperature"), ("tmin", "Min temperature")], strict=True):
        for model, style in styles.items():
            s = scores[(scores["model"] == model) & (scores["target"] == target)].sort_values("horizon")
            ax.plot(s["horizon"], s["mae"], marker="o", ms=4, label=model, **style)
        ax.set_title(label)
        ax.set_xlabel("Days ahead")
        ax.set_xticks(range(1, 8))
    axes[0].set_ylabel("MAE (°C)")
    axes[1].legend(frameon=False, fontsize=8.5)
    fig.suptitle("The model beats both baselines at every horizon", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "error_by_horizon.png")


def backtest_example(df: pd.DataFrame) -> None:
    """A month of real day-ahead forecasts from the test period, with the 80% band."""
    train, _ = evaluate.split(df)
    model = EnsembleForecaster((1,)).fit(train)
    band = ConformalBand().fit(lambda: EnsembleForecaster((1,)), train)
    window = df["2025-01-01":"2025-04-30"]  # 30+ days of history before March
    pred = model.predict(window, 1).shift(1)  # forecast made the day before
    pred = pred["2025-03-01":"2025-03-31"]
    truth = window["2025-03-01":"2025-03-31"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for t, color, label in [("tmax", HOT, "max"), ("tmin", COLD, "min")]:
        w = band.half_width[(t, 1)]
        ax.fill_between(pred.index, pred[t] - w, pred[t] + w, color=color, alpha=0.15, lw=0)
        ax.plot(pred.index, pred[t], color=color, ls="--", lw=1.6, label=f"Forecast {label} (made the day before)")
        ax.plot(truth.index, truth[t], color=color, marker="o", ms=3.5, lw=1.4, label=f"Observed {label}")
    mae = (pred - truth).abs().mean()
    ax.set_title(f"Day-ahead forecasts vs reality, March 2025 (MAE max {mae['tmax']:.1f} °C, min {mae['tmin']:.1f} °C)")
    ax.set_ylabel("°C")
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "backtest_march_2025.png")


def warming(df: pd.DataFrame) -> float:
    complete = df[df.index.year < df.index.max().year]
    yearly = complete.groupby(complete.index.year)["tmax"].mean()
    slope, intercept = np.polyfit(yearly.index, yearly.to_numpy(), 1)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(yearly.index, yearly, marker="o", ms=4, color=HOT, label="Yearly average of daily max")
    ax.plot(
        yearly.index,
        slope * yearly.index + intercept,
        color=GREY,
        ls="--",
        label=f"Trend: {slope * 10:+.2f} °C per decade",
    )
    ax.set_title(f"Marrakech is warming: daily maximums, {yearly.index.min()}–{yearly.index.max()}")
    ax.set_ylabel("°C")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "warming.png")
    return slope * 10


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = data.load(recent=None)
    scores = pd.read_csv(ROOT / "reports" / "scores.csv")
    v1_vs_v2(scores)
    by_horizon(scores)
    backtest_example(df)
    per_decade = warming(df)
    print(f"Figures written to {OUT} (warming trend {per_decade:+.2f} °C per decade)")


if __name__ == "__main__":
    main()
