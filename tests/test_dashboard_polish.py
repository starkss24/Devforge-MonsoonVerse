"""Final polish on the Maps tab: no blank map while painting, and an honest note about what the panel shows."""
import sys
from pathlib import Path

DASH = Path(__file__).resolve().parent.parent / "dashboard"
sys.path.insert(0, str(DASH))          # dashboard modules import each other by bare name, like the other dashboard tests
import tab_maps  # noqa: E402

APP = (DASH / "app.py").read_text(encoding="utf-8")


def test_map_placeholder_while_painting():
    assert "Drawing the map" in tab_maps.PAINTING_HTML and "mm-painting" in tab_maps.PAINTING_HTML
    # the CSS backdrop behind the big map (key maps_main) carries the same message until Plotly paints over it
    assert ".st-key-maps_main" in APP and "Drawing the map" in APP and ".mm-painting" in APP


def test_panel_note_explains_play_vs_selected_day():
    note = tab_maps.PANEL_NOTE_HTML
    assert "selected day" in note and "Play" in note and "info box" in note
    assert ".mm-panel-note" in APP and "font-size: 0.85rem" in APP.split(".mm-panel-note", 1)[1][:80]  # >= 13 px
