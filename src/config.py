from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Sequence

# --------------------------------------------------------------------------------------
# Paths & caching
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
REF_PATH = BASE_DIR / "Ref.xlsx"
PATIENT_CACHE_TTL = 300

# --------------------------------------------------------------------------------------
# Timepoints & columns
# --------------------------------------------------------------------------------------


class TimePoint(Enum):
    T1 = ("Time-point 1", "raw_T1")
    T2 = ("Time-point 2", "raw_T2")
    T3 = ("Time-point 3", "raw_T3")

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def raw_column(self) -> str:
        return self.value[1]


TIMEPOINTS: Sequence[str] = tuple(tp.label for tp in TimePoint)
RAW_COLUMNS = {tp.label: tp.raw_column for tp in TimePoint}

METABOLITE_COLUMN = "Метаболит"
NORMS_COLUMN = "Нормы"
REF_MIN_COLUMN = "ref_min"
REF_MAX_COLUMN = "ref_max"
REQUIRED_SAMPLE_COLUMNS = ("Код", "Группа")
BASE_CODE_COLUMN = "__base_code"

# --------------------------------------------------------------------------------------
# Display constants
# --------------------------------------------------------------------------------------
PLACEHOLDER = "—"
EPS = 1e-12

COLOR_BELOW = "#1976d2"
COLOR_ABOVE = "#d32f2f"
COLOR_NEUTRAL = "#616161"
COLOR_MISSING = "#9e9e9e"
COLOR_BG_BELOW = "#b3d1ff"
COLOR_BG_ABOVE = "#ff9999"
COLOR_BG_NORMAL = "#ffffff"
COLOR_BG_MISSING = "#f2f2f2"

DEFAULT_PADDING_RATIO = 0.1
DEFAULT_FIG_HEIGHT = 540
BASE_LINE_COLOR = "#212121"

TIMEPOINT_COLUMN_WIDTH = 160
AGGRID_COLUMN_WIDTHS = {METABOLITE_COLUMN: 200, NORMS_COLUMN: 160}
AGGRID_DEFAULT_HEIGHT = 520
AGGRID_THEME = "balham"

COMPARISON_BUTTON_LABEL = "➕ Добавить пациента для сравнения"

# --------------------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------------------
GROUP_KEYWORDS: dict[TimePoint, tuple[str, ...]] = {
    TimePoint.T1: ("до",),
    TimePoint.T2: ("после",),
    TimePoint.T3: ("след", "повтор"),
}
SUFFIX_TO_TIMEPOINT = {"1": TimePoint.T1, "2": TimePoint.T2, "3": TimePoint.T3}
CODE_SUFFIX_PATTERN = re.compile(r"-(\d+)\s*(?:\([a-z]+\))?$", re.IGNORECASE)
BASE_CODE_SUFFIX_PATTERN = re.compile(r"-\d+[A-Za-z()]*$")

TP_JS_MAP_JSON = json.dumps(RAW_COLUMNS)

# --------------------------------------------------------------------------------------
# AgGrid styling
# --------------------------------------------------------------------------------------
SELECTION_OVERLAY = "inset 0 0 0 9999px rgba(204,229,255,0.45)"
TIMEPOINT_CELL_STYLE_JSON = json.dumps(
    {"border": "1px solid #666", "textAlign": "center", "fontSize": "13px"}
)
METABOLITE_CELL_STYLE_JSON = json.dumps(
    {"border": "1px solid #666", "textAlign": "left", "fontSize": "13px", "cursor": "pointer"}
)
NORMS_CELL_STYLE_JSON = json.dumps(
    {"border": "1px solid #666", "textAlign": "center", "fontSize": "13px", "fontStyle": "italic"}
)

AGGRID_CUSTOM_CSS = """
<style>
  .ag-theme-balham { font-size:13px; }
  .ag-theme-balham .ag-cell {
      padding-top:5px !important;
      padding-bottom:5px !important;
  }
  .ag-theme-balham .ag-header-cell-label {
      font-size:13px;
      white-space:normal !important;
      line-height:1.2;
  }
</style>
"""

AGGRID_ON_CELL_CLICK_JS = """
function(params){
  const api = params.api, node = params.node, was = node.isSelected();
  api.deselectAll();
  if (!was) node.setSelected(true);
  api.refreshCells({ force: true });
}
"""

# --------------------------------------------------------------------------------------
# Sorting modes
# --------------------------------------------------------------------------------------


class SortMode(str, Enum):
    ALPHA = "По алфавиту"
    BELOW_FIRST = "Сначала ниже нормы"
    ABOVE_FIRST = "Сначала выше нормы"


SORT_MODES: Sequence[str] = tuple(mode.value for mode in SortMode)
