"""Builds notebooks/exploration.ipynb (then run it with nbconvert --execute)."""

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Forecasting Marrakech temperatures — exploration

This notebook walks through the data and the modelling choices behind the `tempforecast` package.
Run it from the project root after `pip install -e ".[notebook]"`."""),
    code("""import matplotlib.pyplot as plt
import pandas as pd

from tempforecast import data, evaluate, forecast

plt.rcParams.update({"figure.figsize": (10, 3.5), "axes.grid": True, "grid.alpha": 0.3})
df = data.load(recent=None)
print(f"{len(df):,} days from {df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}")
df.tail()"""),
    md("""## 1. Why the first version could not work

Version 1 predicted the temperature from the **day of the week**. If the weekday mattered, the
temperatures would differ from Monday to Sunday. They do not:"""),
    code("""weekday = df.groupby(df.index.day_name())["tmax"].mean()
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ax = weekday.loc[order].plot(kind="bar", color="#e8590c", ylim=(20, 30), rot=0,
                             title="Average daily maximum by weekday, 2000–2026")
ax.set_ylabel("°C");"""),
    md("""The averages differ by a few tenths of a degree: the weekday carries no information about the weather.
What does carry information is **the season** and **the weather of the last days**."""),
    code("""fig, ax = plt.subplots()
doy = df.groupby(df.index.day_of_year)[["tmax", "tmin"]].mean()
doy.plot(ax=ax, color=["#e8590c", "#1c7ed6"], title="Average temperature by day of the year")
ax.set_xlabel("Day of the year")
ax.set_ylabel("°C");"""),
    code("""anomaly = df["tmax"] - df["tmax"].groupby(df.index.day_of_year).transform("mean")
lags = range(1, 15)
title = "How long does a warm or cold spell last? (autocorrelation of the anomaly)"
pd.Series([anomaly.autocorr(lag) for lag in lags], index=lags).plot(kind="bar", rot=0, color="#495057", title=title)
plt.xlabel("Days later");"""),
    md("""An unusually warm day is usually followed by another warm day, and the link fades within a week:
this is why recent temperatures help for the next few days, and why the 7-day forecast is harder."""),
    md("""## 2. Honest evaluation

Models are trained on 2000–2022 and tested on 2023 onwards. Version 1 shuffled the days before splitting,
which lets a model train on days that come after the ones it is tested on."""),
    code("""# Produced by: tempforecast evaluate --output reports/scores.csv --coverage reports/coverage.csv
scores = pd.read_csv("../reports/scores.csv")
evaluate.summary(scores, horizon=1).round(2)"""),
    code("""pivot = scores[scores["target"] == "tmax"].pivot(index="horizon", columns="model", values="mae")
pivot[["Persistence", "Climatology", "Ensemble (ridge + boosting)"]].plot(
    marker="o", title="Error on the max temperature by horizon (°C)");"""),
    code("""pd.read_csv("../reports/coverage.csv")"""),
    md("""The 80% band (conformal prediction) contains the real temperature about 80% of the time on the test
years, so the uncertainty shown in the dashboard can be trusted."""),
    md("""## 3. The next 7 days"""),
    code("""trained = forecast.train(df)
forecast.next_days(trained, df)"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, "notebooks/exploration.ipynb")
