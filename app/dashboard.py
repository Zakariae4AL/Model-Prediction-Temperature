"""Streamlit dashboard: streamlit run app/dashboard.py"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from tempforecast import data, evaluate, forecast
from tempforecast.models import Climatology

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
HOT, COLD = "#ff8c42", "#4dabf7"

TEXT = {
    "en": {
        "title": "Marrakech · 7-day temperature forecast",
        "caption": "Machine learning on {days:,} days of real weather (Open-Meteo / ERA5, {start}–{end}).",
        "tab_forecast": "Forecast",
        "tab_models": "Model comparison",
        "tab_history": "History",
        "tab_how": "How it works",
        "tomorrow": "Tomorrow",
        "max": "Max",
        "min": "Min",
        "band": "80% range: {low}–{high} °C",
        "chart_title": "Last 30 days and the next 7",
        "observed": "Observed",
        "forecast": "Forecast",
        "day": "Day",
        "provisional": "The last {n} day(s) are provisional (recent analysis, replaced by the archive after ~5 days).",
        "from_last": "Forecast made from the last observed day: {day}.",
        "update": "Update data",
        "updating": "Downloading the latest weather from Open-Meteo…",
        "updated": "Data updated.",
        "update_failed": "Could not download data: {error}",
        "training": "Training the models on all available days (first load only)…",
        "mae_title": "Error on tomorrow's forecast (MAE, °C, lower is better)",
        "horizon_title": "How the error grows with the forecast horizon",
        "horizon": "Days ahead",
        "model": "Model",
        "target": "Temperature",
        "coverage_title": "Is the 80% range honest?",
        "coverage_text": "Share of test days where the real temperature fell inside the predicted range.",
        "v1_vs_v2": "The v1 model (weekday → temperature) was off by **{v1:.1f} °C** on average; "
        "the new model is off by **{v2:.1f} °C**, {x:.1f}× more accurate.",
        "test_note": "All models trained on {train} and tested on {test}, days the models never saw.",
        "year": "Year",
        "normal": "Normal for the date",
        "yearly_title": "Average daily maximum per year",
        "how": """
**Data.** Daily weather for Marrakech from the Open-Meteo archive (ERA5 reanalysis): max and min
temperature, rain, wind speed and direction, solar radiation and sunshine.

**Features.** For each day: the temperatures of the last days, 3/7/30-day averages, recent change,
rain, wind (direction encoded as an angle), sunshine, the season (sine/cosine of the day of the year)
and the normal temperature of the date being forecast. Every feature only uses the past.

**Models.** Two baselines (same as today; the normal for the date) and three learned models (ridge
regression, gradient boosting and their average). One model per target and per horizon (1 to 7 days).

**Validation.** Chronological split: trained on 2000–2022, tested on 2023 onwards. The v1 notebook
shuffled days at random, which lets a model learn from days that come after the ones it is tested on.

**Uncertainty.** The 80% range comes from conformal prediction: errors are measured on the last
three years before the test period and their 80th percentile becomes the band.
""",
    },
    "fr": {
        "title": "Marrakech · prévision de température à 7 jours",
        "caption": "Apprentissage automatique sur {days:,} jours de météo réelle (Open-Meteo / ERA5, {start}–{end}).",
        "tab_forecast": "Prévision",
        "tab_models": "Comparaison des modèles",
        "tab_history": "Historique",
        "tab_how": "Fonctionnement",
        "tomorrow": "Demain",
        "max": "Max",
        "min": "Min",
        "band": "Intervalle à 80 % : {low}–{high} °C",
        "chart_title": "Les 30 derniers jours et les 7 suivants",
        "observed": "Observé",
        "forecast": "Prévision",
        "day": "Jour",
        "provisional": "Les {n} dernier(s) jour(s) sont provisoires "
        "(analyse récente, remplacée par l’archive après ~5 jours).",
        "from_last": "Prévision faite à partir du dernier jour observé : {day}.",
        "update": "Mettre à jour les données",
        "updating": "Téléchargement de la météo récente depuis Open-Meteo…",
        "updated": "Données mises à jour.",
        "update_failed": "Téléchargement impossible : {error}",
        "training": "Entraînement des modèles sur toutes les données (premier chargement seulement)…",
        "mae_title": "Erreur sur la prévision de demain (MAE, °C, plus bas = mieux)",
        "horizon_title": "Comment l’erreur augmente avec l’échéance",
        "horizon": "Jours à l’avance",
        "model": "Modèle",
        "target": "Température",
        "coverage_title": "L’intervalle à 80 % est-il fiable ?",
        "coverage_text": "Part des jours de test où la température réelle est tombée dans l’intervalle prévu.",
        "v1_vs_v2": "Le modèle v1 (jour de la semaine → température) se trompait de **{v1:.1f} °C** en moyenne ; "
        "le nouveau modèle de **{v2:.1f} °C**, soit {x:.1f}× plus précis.",
        "test_note": "Tous les modèles sont entraînés sur {train} et testés sur {test}, des jours jamais vus.",
        "year": "Année",
        "normal": "Normale de saison",
        "yearly_title": "Maximum journalier moyen par année",
        "how": """
**Données.** Météo quotidienne de Marrakech issue de l’archive Open-Meteo (réanalyse ERA5) :
températures max et min, pluie, vent (vitesse et direction), rayonnement solaire et ensoleillement.

**Variables.** Pour chaque jour : les températures des derniers jours, les moyennes sur 3, 7 et 30 jours,
la tendance récente, la pluie, le vent (direction codée comme un angle), l’ensoleillement, la saison
(sinus/cosinus du jour de l’année) et la normale du jour à prévoir. Chaque variable n’utilise que le passé.

**Modèles.** Deux références (comme aujourd’hui ; la normale de saison) et trois modèles appris
(régression ridge, gradient boosting et leur moyenne). Un modèle par cible et par échéance (1 à 7 jours).

**Validation.** Découpage chronologique : entraînement sur 2000–2022, test à partir de 2023. Le notebook v1
mélangeait les jours au hasard, ce qui permet d’apprendre sur des jours postérieurs à ceux du test.

**Incertitude.** L’intervalle à 80 % vient de la prédiction conforme : les erreurs sont mesurées sur les
trois dernières années avant la période de test et leur 80e centile donne la largeur de l’intervalle.
""",
    },
}

st.set_page_config(page_title="Marrakech temperature forecast", page_icon="🌡️", layout="wide")


@st.cache_data(show_spinner=False)
def load_data(stamp: float) -> pd.DataFrame:
    return data.load()


@st.cache_resource(show_spinner=False)
def train_model(last_day: str, rows: int):
    return forecast.train(load_data(_stamp()))


@st.cache_data(show_spinner=False)
def load_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    scores_path, coverage_path = REPORTS / "scores.csv", REPORTS / "coverage.csv"
    if scores_path.exists() and coverage_path.exists():
        return pd.read_csv(scores_path), pd.read_csv(coverage_path)
    df = load_data(_stamp())
    return evaluate.evaluate(df), evaluate.interval_coverage(df)


def _stamp() -> float:
    """Changes when the data files change, so cached data is reloaded."""
    return sum(p.stat().st_mtime for p in (data.DEFAULT_DATA_PATH, data.RECENT_DATA_PATH) if p.exists())


lang = st.sidebar.radio("Language / Langue", ["en", "fr"], format_func=lambda c: {"en": "English", "fr": "Français"}[c])
T = TEXT[lang]
alt.data_transformers.disable_max_rows()

if st.sidebar.button(T["update"]):
    with st.spinner(T["updating"]):
        try:
            data.fetch()
            data.fetch_recent()
            st.cache_data.clear()
            st.sidebar.success(T["updated"])
        except OSError as error:
            st.sidebar.error(T["update_failed"].format(error=error))

df = load_data(_stamp())
st.title(T["title"])
st.caption(T["caption"].format(days=len(df), start=df.index.min().year, end=df.index.max().year))

tab_forecast, tab_models, tab_history, tab_how = st.tabs(
    [T["tab_forecast"], T["tab_models"], T["tab_history"], T["tab_how"]]
)

# ---------------- Forecast ----------------
with tab_forecast:
    with st.spinner(T["training"]):
        trained = train_model(str(df.index.max().date()), len(df))
    table = forecast.next_days(trained, df)
    first = table.iloc[0]

    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric(f"{T['tomorrow']} · {T['max']}", f"{first.tmax:.1f} °C")
    c1.caption(T["band"].format(low=first.tmax_low, high=first.tmax_high))
    c2.metric(f"{T['tomorrow']} · {T['min']}", f"{first.tmin:.1f} °C")
    c2.caption(T["band"].format(low=first.tmin_low, high=first.tmin_high))
    with c3:
        st.caption(T["from_last"].format(day=f"{df.index.max():%Y-%m-%d}"))
        provisional = int(df["provisional"].sum())
        if provisional:
            st.caption(T["provisional"].format(n=provisional))

    recent = df.iloc[-30:][["tmax", "tmin"]].reset_index()
    observed = recent.melt("date", var_name="kind", value_name="value")
    observed["series"] = T["observed"]
    future = table.reset_index()
    predicted = future.melt("date", ["tmax", "tmin"], var_name="kind", value_name="value")
    predicted["series"] = T["forecast"]
    bands = pd.concat(
        [
            future[["date", "tmax_low", "tmax_high"]]
            .rename(columns={"tmax_low": "low", "tmax_high": "high"})
            .assign(kind="tmax"),
            future[["date", "tmin_low", "tmin_high"]]
            .rename(columns={"tmin_low": "low", "tmin_high": "high"})
            .assign(kind="tmin"),
        ]
    )
    color = alt.Color(
        "kind:N",
        scale=alt.Scale(domain=["tmax", "tmin"], range=[HOT, COLD]),
        legend=alt.Legend(title=None, labelExpr="datum.label == 'tmax' ? 'Max' : 'Min'"),
    )
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%d %b"))
    band_chart = alt.Chart(bands).mark_area(opacity=0.22).encode(x=x, y="low:Q", y2="high:Q", color=color)
    lines = (
        alt.Chart(pd.concat([observed, predicted]))
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=x,
            y=alt.Y("value:Q", title="°C", scale=alt.Scale(zero=False)),
            color=color,
            strokeDash=alt.StrokeDash("series:N", legend=alt.Legend(title=None), sort=[T["observed"], T["forecast"]]),
            tooltip=[
                alt.Tooltip("date:T", format="%a %d %b"),
                "series:N",
                "kind:N",
                alt.Tooltip("value:Q", format=".1f"),
            ],
        )
    )
    st.subheader(T["chart_title"])
    st.altair_chart((band_chart + lines).properties(height=380), use_container_width=True)

    shown = table.copy()
    shown.index = [f"{d:%a %d %b}" for d in shown.index]
    shown = shown.drop(columns="horizon")
    shown.columns = ["Max", "Max ↓", "Max ↑", "Min", "Min ↓", "Min ↑"]
    st.dataframe(shown.style.format("{:.1f}"), use_container_width=True)

# ---------------- Models ----------------
with tab_models:
    scores, coverage = load_scores()
    day1 = scores[scores["horizon"] == 1]
    mae = day1.pivot(index="model", columns="target", values="mae")
    v1 = mae.loc["v1 · weekday regression", "tmax"]
    v2 = mae.loc["Ensemble (ridge + boosting)", "tmax"]
    st.info(T["v1_vs_v2"].format(v1=v1, v2=v2, x=v1 / v2))
    st.caption(
        T["test_note"].format(
            train=f"2000–{int(evaluate.TEST_START[:4]) - 1}", test=f"{evaluate.TEST_START[:4]}–{df.index.max().year}"
        )
    )

    st.subheader(T["mae_title"])
    bars = (
        alt.Chart(day1)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("model:N", sort=alt.EncodingSortField("mae", op="mean"), title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("mae:Q", title="MAE (°C)"),
            color=alt.Color(
                "target:N",
                scale=alt.Scale(domain=["tmax", "tmin"], range=[HOT, COLD]),
                legend=alt.Legend(title=T["target"], labelExpr="datum.label == 'tmax' ? 'Max' : 'Min'"),
            ),
            yOffset="target:N",
            tooltip=["model", "target", alt.Tooltip("mae:Q", format=".2f"), alt.Tooltip("within_2c:Q", format=".0%")],
        )
        .properties(height=340)
    )
    st.altair_chart(bars, use_container_width=True)

    st.subheader(T["horizon_title"])
    target = st.radio(
        T["target"], ["tmax", "tmin"], horizontal=True, format_func=lambda t: {"tmax": "Max", "tmin": "Min"}[t]
    )
    keep = ["Persistence", "Climatology", "Ridge regression", "Gradient boosting", "Ensemble (ridge + boosting)"]
    curve = scores[(scores["target"] == target) & scores["model"].isin(keep)]
    st.altair_chart(
        alt.Chart(curve)
        .mark_line(point=True)
        .encode(
            x=alt.X("horizon:O", title=T["horizon"]),
            y=alt.Y("mae:Q", title="MAE (°C)"),
            color=alt.Color("model:N", title=T["model"]),
            tooltip=["model", "horizon", alt.Tooltip("mae:Q", format=".2f")],
        )
        .properties(height=320),
        use_container_width=True,
    )

    st.subheader(T["coverage_title"])
    st.caption(T["coverage_text"])
    cov = coverage.copy()
    cov["coverage"] = (cov["coverage"] * 100).round(1).astype(str) + " %"
    cov.columns = [T["target"], T["horizon"], "Coverage", "Band"]
    st.dataframe(cov, hide_index=True, use_container_width=True)

# ---------------- History ----------------
with tab_history:
    years = sorted(df.index.year.unique())
    year = st.select_slider(T["year"], options=years, value=years[-1])
    one = df[df.index.year == year][["tmax", "tmin"]]
    normal = Climatology().fit(df).for_dates(one.index)
    long = one.reset_index().melt("date", var_name="kind", value_name="value")
    norm = normal.reset_index(names="date").melt("date", var_name="kind", value_name="value")
    color = alt.Color("kind:N", scale=alt.Scale(domain=["tmax", "tmin"], range=[HOT, COLD]), legend=None)
    actual = (
        alt.Chart(long)
        .mark_line(strokeWidth=1.4)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("value:Q", title="°C"),
            color=color,
            tooltip=[alt.Tooltip("date:T", format="%d %b %Y"), "kind", alt.Tooltip("value:Q", format=".1f")],
        )
    )
    normal_line = (
        alt.Chart(norm)
        .mark_line(strokeDash=[4, 4], opacity=0.7, color="#9aa0b4")
        .encode(x="date:T", y="value:Q", detail="kind:N")
    )
    st.altair_chart((normal_line + actual).properties(height=360), use_container_width=True)
    st.caption(f"┄┄ {T['normal']}")

    st.subheader(T["yearly_title"])
    complete = df[df.index.year < df.index.max().year]  # skip the unfinished current year
    yearly = complete.groupby(complete.index.year)["tmax"].mean().rename_axis("year").reset_index(name="value")
    base = alt.Chart(yearly).encode(
        x=alt.X("year:Q", title=None, axis=alt.Axis(format="d")),
        y=alt.Y("value:Q", title="°C", scale=alt.Scale(zero=False)),
    )
    points = base.mark_line(point=alt.OverlayMarkDef(color=HOT), color=HOT).encode(
        tooltip=["year", alt.Tooltip("value:Q", format=".2f")]
    )
    trend = base.transform_regression("year", "value").mark_line(strokeDash=[6, 4], color="#9aa0b4")
    st.altair_chart((points + trend).properties(height=300), use_container_width=True)

# ---------------- How it works ----------------
with tab_how:
    st.markdown(T["how"])
