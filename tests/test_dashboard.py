from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")


@pytest.mark.parametrize("language", ["English", "Français"])
def test_dashboard_runs_without_errors(language):
    app = AppTest.from_file(APP, default_timeout=600).run()
    if language != "English":
        app.sidebar.radio[0].set_value("fr").run()
    assert not app.exception, app.exception
    assert len(app.tabs) == 4
    metrics = [m.label for m in app.metric]
    assert any("Max" in label for label in metrics) and any("Min" in label for label in metrics)
    assert app.title[0].value.startswith("Marrakech")
