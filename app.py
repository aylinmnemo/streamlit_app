from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence
import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative as plotly_qual
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# Constants and basic settings
BASE_DIR = Path(__file__).resolve().parent
REF_PATH = BASE_DIR / "Ref.xlsx"


class TimePoint(Enum):
    """Точки измерения с понятной подписью и колонкой сырых значений."""

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

PLACEHOLDER = "—"
EPS = 1e-12
TIMEPOINT_COLUMN_WIDTH = 160
AGGRID_COLUMN_WIDTHS = {METABOLITE_COLUMN: 200, NORMS_COLUMN: 160}
AGGRID_DEFAULT_HEIGHT = 520
AGGRID_THEME = "balham"
BASE_LINE_COLOR = "#212121"
DEFAULT_PADDING_RATIO = 0.1
DEFAULT_FIG_HEIGHT = 540
COMPARISON_BUTTON_LABEL = "➕ Добавить пациента для сравнения"
COLOR_BELOW = "#1976d2"
COLOR_ABOVE = "#d32f2f"
COLOR_NEUTRAL = "#616161"
COLOR_MISSING = "#9e9e9e"


class SortMode(str, Enum):
    ALPHA = "По алфавиту"
    BELOW_FIRST = "Сначала ниже нормы"
    ABOVE_FIRST = "Сначала выше нормы"


SORT_MODES: Sequence[str] = tuple(mode.value for mode in SortMode)

GROUP_KEYWORDS: dict[TimePoint, tuple[str, ...]] = {
    TimePoint.T1: ("до",),
    TimePoint.T2: ("после",),
    TimePoint.T3: ("след", "повтор"),
}
SUFFIX_TO_TIMEPOINT = {"1": TimePoint.T1, "2": TimePoint.T2, "3": TimePoint.T3}
CODE_SUFFIX_PATTERN = re.compile(r"-(\d+)\s*(?:\([a-z]+\))?$")
BASE_CODE_SUFFIX_PATTERN = re.compile(r"-\d+[A-Za-z()]*$")
TP_JS_MAP_JSON = json.dumps(RAW_COLUMNS)

TIMEPOINT_CELL_STYLE_JSON = json.dumps(
    {"border": "1px solid #666", "textAlign": "center", "fontSize": "13px"}
)
METABOLITE_CELL_STYLE_JSON = json.dumps(
    {"border": "1px solid #666", "textAlign": "left", "fontSize": "13px", "cursor": "pointer"}
)
NORMS_CELL_STYLE_JSON = json.dumps(
    {"border": "1px solid #666", "textAlign": "center", "fontSize": "13px", "fontStyle": "italic"}
)
SELECTION_OVERLAY = "inset 0 0 0 9999px rgba(204,229,255,0.45)"

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


# Utilities
def to_float(value: Any) -> float:
    """Мягкое преобразование к float с поддержкой запятой и пустых значений."""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return float("nan")
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return float("nan")


def session_key(*parts: Iterable[Any]) -> str:
    """Стабильный ключ для состояния Streamlit."""
    return "__".join(str(part) for part in parts if part is not None and part != "")


def resolve_tp(code: str, group: str) -> str:
    """Определить таймпоинт по текстовым подсказкам и суффиксам в коде пациента."""
    group_text = str(group).lower()
    code_text = str(code).lower()
    for tp, keywords in GROUP_KEYWORDS.items():
        if any(keyword in group_text for keyword in keywords):
            if tp is TimePoint.T2 and "-2" in code_text and "(t)" in code_text:
                return TimePoint.T3.label
            return tp.label

    suffix_match = CODE_SUFFIX_PATTERN.search(code_text)
    if suffix_match:
        suffix = suffix_match.group(1)
        if suffix == "2" and "(t)" in code_text:
            return TimePoint.T3.label
        matched_tp = SUFFIX_TO_TIMEPOINT.get(suffix)
        if matched_tp:
            return matched_tp.label

    for digit, tp in SUFFIX_TO_TIMEPOINT.items():
        if re.search(fr"-{digit}(?!\d)", code_text):
            return tp.label
    return ""


def display_value_precise(
    value: Any,
    ref_min_value: Any,
    ref_max_value: Any,
    max_extra_decimals: int = 6,
) -> str:
    """Адаптивное форматирование числа рядом с референсными границами."""
    v, mn, mx = to_float(value), to_float(ref_min_value), to_float(ref_max_value)
    if any(pd.isna(x) for x in (v, mn, mx)):
        return ""

    def decimals(number: float, max_decimals: int = 8) -> int:
        text = f"{float(number):.{max_decimals}f}".rstrip("0").rstrip(".")
        return len(text.split(".")[1]) if "." in text else 0

    base = max(decimals(mn), decimals(mx))
    is_low, is_high = v < mn - EPS, v > mx + EPS

    for extra in range(max_extra_decimals + 1):
        digits = base + extra
        formatted = f"{v:.{digits}f}"
        try:
            rounded = float(formatted)
        except ValueError:
            continue
        if rounded == 0.0 and abs(v) > 0.0:
            continue
        masked = (is_low and rounded >= mn) or (is_high and rounded <= mx)
        if not masked:
            return formatted.rstrip("0").rstrip(".")

    return f"{v:.{base + max_extra_decimals}f}".rstrip("0").rstrip(".")


def lookup_reference(reference: pd.Series, metabolite: str) -> float:
    """Найти значение референса, игнорируя возможные пробелы и регистр."""
    if reference.empty:
        return float("nan")

    candidates = [str(metabolite), str(metabolite).strip()]
    for candidate in candidates:
        if candidate in reference.index:
            return to_float(reference[candidate])

    metabolite_lower = str(metabolite).strip().lower()
    for idx in reference.index:
        if str(idx).strip().lower() == metabolite_lower:
            return to_float(reference[idx])

    return float("nan")


def reference_bounds(
    ref_min_series: pd.Series,
    ref_max_series: pd.Series,
    record: pd.Series,
    metabolite: str,
) -> tuple[float, float]:
    """Получить нижнюю и верхнюю границу для метаболита, предпочитая данные из таблицы пациента."""
    rmin = to_float(record.get(REF_MIN_COLUMN, float("nan")))
    rmax = to_float(record.get(REF_MAX_COLUMN, float("nan")))
    if pd.isna(rmin):
        rmin = lookup_reference(ref_min_series, metabolite)
    if pd.isna(rmax):
        rmax = lookup_reference(ref_max_series, metabolite)
    return rmin, rmax


# Model
@dataclass(frozen=True)
class MetaboliteRange:
    """Диапазон нормы для метаболита."""

    name: str
    mn: float
    mx: float

    def norm_text(self) -> str:
        return f"{self.mn:g} - {self.mx:g}"


@dataclass
class PatientSnapshot:
    """Контекст по пациенту: строки исходных данных и диапазоны норм."""

    code: str
    rows: pd.DataFrame
    ranges: dict[str, MetaboliteRange]

    @property
    def metabolites(self) -> list[str]:
        return [metabolite for metabolite in self.ranges if metabolite in self.rows.columns]

    def table(self) -> pd.DataFrame:
        """Сформировать таблицу пациента с нормами и таймпоинтами."""
        empty_columns = [METABOLITE_COLUMN, NORMS_COLUMN, *TIMEPOINTS, REF_MIN_COLUMN, REF_MAX_COLUMN]
        if self.rows.empty:
            return pd.DataFrame(columns=empty_columns)

        metabolite_names = self.metabolites
        if not metabolite_names:
            return pd.DataFrame(columns=empty_columns)

        assigned = self.rows.assign(
            tp=lambda frame: frame.apply(lambda row: resolve_tp(row["Код"], row["Группа"]), axis=1)
        )
        tidy_source = assigned[assigned["tp"].isin(TIMEPOINTS)]

        if tidy_source.empty:
            pivot = pd.DataFrame(index=metabolite_names, columns=TIMEPOINTS, dtype=float)
        else:
            tidy = (
                tidy_source.melt(
                    id_vars=["tp"],
                    value_vars=metabolite_names,
                    var_name=METABOLITE_COLUMN,
                    value_name="value",
                )
                .dropna(subset=["value"])
            )
            if tidy.empty:
                pivot = pd.DataFrame(index=metabolite_names, columns=TIMEPOINTS, dtype=float)
            else:
                pivot = (
                    tidy.pivot_table(index=METABOLITE_COLUMN, columns="tp", values="value", aggfunc="last")
                    .reindex(columns=TIMEPOINTS)
                )

        pivot = pivot.reindex(index=metabolite_names, columns=TIMEPOINTS)
        pivot.columns.name = None

        base = pd.DataFrame(
            {
                METABOLITE_COLUMN: metabolite_names,
                REF_MIN_COLUMN: [self.ranges[name].mn for name in metabolite_names],
                REF_MAX_COLUMN: [self.ranges[name].mx for name in metabolite_names],
                NORMS_COLUMN: [self.ranges[name].norm_text() for name in metabolite_names],
            }
        ).set_index(METABOLITE_COLUMN)

        combined = base.join(pivot)
        return combined.reset_index()[[METABOLITE_COLUMN, NORMS_COLUMN, *TIMEPOINTS, REF_MIN_COLUMN, REF_MAX_COLUMN]]


# Uploading data
@st.cache_data(show_spinner=False, ttl=300)
def load_reference_data(path: Path) -> tuple[pd.Series, pd.Series, dict[str, list[str]]]:
    """Загрузить диапазоны норм и карту рисков из Ref.xlsx."""
    try:
        ref = pd.read_excel(path, sheet_name=0)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(f"Не удалось прочитать лист 1: {exc}") from exc

    if "metabolite" not in ref.columns:
        raise ValueError("На листе 1 должна быть колонка 'metabolite'.")

    def pick_row(name: str) -> pd.Series:
        mask = ref["metabolite"].astype(str).str.strip().str.lower() == name
        if not mask.any():
            raise ValueError(f"В листе 1 отсутствует строка '{name}'.")
        row = ref.loc[mask].iloc[0, 1:]
        if row.isnull().all():
            raise ValueError(f"Пустые ref_{name[4:]} на листе 1.")
        return row.apply(to_float)

    ref_min = pick_row("ref_min")
    ref_max = pick_row("ref_max")

    try:
        risk_ref = pd.read_excel(path, sheet_name=1)
    except ValueError:
        risk_ref = pd.DataFrame()
    except Exception as exc:
        raise ValueError(f"Не удалось прочитать лист 2: {exc}") from exc

    risk_map: dict[str, list[str]] = {}
    if not risk_ref.empty and {"Группа_риска", "Маркер / Соотношение"}.issubset(risk_ref.columns):
        for _, row in risk_ref.iterrows():
            group = str(row["Группа_риска"]).strip()
            metabolite = str(row["Маркер / Соотношение"]).strip()
            if group and metabolite and group.lower() != "nan" and metabolite.lower() != "nan":
                risk_map.setdefault(group, []).append(metabolite)
        for key, values in risk_map.items():
            risk_map[key] = sorted(set(values))

    return ref_min, ref_max, risk_map


def load_sample(uploaded_file: Any) -> pd.DataFrame:
    """Прочитать файл выгрузки по пациентам и нормализовать значения."""
    try:
        df = pd.read_excel(uploaded_file).copy()
    except Exception as exc:
        raise ValueError(f"Ошибка чтения файла: {exc}") from exc

    missing = set(REQUIRED_SAMPLE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют необходимые колонки: {', '.join(sorted(missing))}")

    numeric_columns = [column for column in df.columns if column not in REQUIRED_SAMPLE_COLUMNS]
    if numeric_columns:
        converted = {column: df[column].map(to_float) for column in numeric_columns}
        df = df.assign(**converted)
        df = df.copy()  # defragment

    df["Код"] = df["Код"].astype(str).str.strip()
    df[BASE_CODE_COLUMN] = df["Код"].apply(_base_patient_code)
    return df


def _base_patient_code(code: Any) -> str:
    """Нормализовать идентификатор пациента, убрав технические суффиксы."""
    text = str(code).strip()
    if not text or text.lower() == "nan":
        return ""
    if text.count("-") >= 2:
        return BASE_CODE_SUFFIX_PATTERN.sub("", text)
    return text


def build_base_df(sample_df: pd.DataFrame, base_code: str, ref_min: pd.Series, ref_max: pd.Series) -> pd.DataFrame:
    """Построить таблицу пациента с нормами и измерениями по таймпоинтам."""
    if BASE_CODE_COLUMN in sample_df.columns:
        rows = sample_df[sample_df[BASE_CODE_COLUMN] == base_code].copy()
    else:
        rows = sample_df[sample_df["Код"].astype(str).str.contains(fr"\b{base_code}\b", case=False, na=False)].copy()
    excluded_columns = set(REQUIRED_SAMPLE_COLUMNS) | {BASE_CODE_COLUMN}
    available_metabolites = [
        column for column in rows.columns if column not in excluded_columns
    ]
    selected = [name for name in available_metabolites if name in ref_min.index and name in ref_max.index]
    filtered_min = ref_min.reindex(selected)
    filtered_max = ref_max.reindex(selected)
    ranges = {
        name: MetaboliteRange(name, filtered_min[name], filtered_max[name])
        for name in selected
    }
    snapshot = PatientSnapshot(base_code, rows, ranges)
    return snapshot.table()


# Analytics and sorting
def add_deviation_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Добавить счётчики отклонений выше/ниже нормы."""
    enriched = df.copy()
    values = enriched.loc[:, TIMEPOINTS]
    below_threshold = enriched[REF_MIN_COLUMN] - EPS
    above_threshold = enriched[REF_MAX_COLUMN] + EPS
    enriched["_below_cnt"] = values.lt(below_threshold, axis=0).sum(axis=1)
    enriched["_above_cnt"] = values.gt(above_threshold, axis=0).sum(axis=1)
    return enriched


def sort_by_status(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Отсортировать таблицу по выбранному режиму."""
    enriched = add_deviation_counts(df)
    if mode == SortMode.ALPHA.value:
        return enriched.sort_values(METABOLITE_COLUMN, ascending=True, ignore_index=True).drop(columns=["_below_cnt", "_above_cnt"])

    if mode not in {SortMode.BELOW_FIRST.value, SortMode.ABOVE_FIRST.value}:
        return enriched.drop(columns=["_below_cnt", "_above_cnt"])

    below = enriched["_below_cnt"]
    above = enriched["_above_cnt"]

    if mode == SortMode.BELOW_FIRST.value:
        status_labels = np.select([below > 0, above > 0], ["below", "above"], default="normal")
        secondary_keys = (-below, -above)
        status_order = ["below", "above", "normal"]
    else:
        status_labels = np.select([above > 0, below > 0], ["above", "below"], default="normal")
        secondary_keys = (-above, -below)
        status_order = ["above", "below", "normal"]

    enriched["_status"] = pd.Categorical(status_labels, categories=status_order, ordered=True)
    enriched["_secondary1"], enriched["_secondary2"] = secondary_keys
    sorted_df = enriched.sort_values(
        ["_status", "_secondary1", "_secondary2", METABOLITE_COLUMN],
        ignore_index=True,
    )
    return sorted_df.drop(columns=["_below_cnt", "_above_cnt", "_status", "_secondary1", "_secondary2"])


# Visualization of tables and graphs
def _js_cell_style(style_json: str) -> JsCode:
    return JsCode(
        f"""
    function(params){{
      const style = {style_json};
      if (params.node && params.node.isSelected()) {{
        style.backgroundColor = '#cce5ff';
      }}
      return style;
    }}
    """
    )


def _color_cells_js() -> JsCode:
    return JsCode(
        f"""
    function(params){{
      const mapping = {TP_JS_MAP_JSON};
      const style = {TIMEPOINT_CELL_STYLE_JSON};
      const rmin = Number(params.data.{REF_MIN_COLUMN});
      const rmax = Number(params.data.{REF_MAX_COLUMN});
      const rawKey = mapping[params.colDef.field];
      let baseColor = '#ffffff';

      let rawValue = rawKey ? params.data[rawKey] : null;
      if (!(rawValue === null || rawValue === undefined || rawValue === '' ||
            rawValue === 'NaN' || rawValue === 'nan')) {{
        const raw = Number(rawValue);
        if (!Number.isNaN(raw)) {{
          if (raw < rmin - {EPS}) baseColor = '#b3d1ff';
          else if (raw > rmax + {EPS}) baseColor = '#ff9999';
        }}
      }}

      style.backgroundColor = baseColor;
      if (params.node && params.node.isSelected()) {{
        style.boxShadow = '{SELECTION_OVERLAY}';
      }}
      return style;
    }}
    """
    )


def render_table_controls(key_prefix: str) -> tuple[str, bool]:
    """Отрисовать контролы перед таблицей."""
    left_caption, right_sort, toggle_col = st.columns([5, 2, 2])
    with left_caption:
        st.markdown(
            '<p style="font-size:14px; font-weight:600;">Нажмите на метаболит — график справа.</p>',
            unsafe_allow_html=True,
        )
    with right_sort:
        sort_mode = st.selectbox(
            "Сортировка:",
            SORT_MODES,
            index=0,
            label_visibility="collapsed",
            key=session_key(key_prefix, "sort_mode"),
        )
    with toggle_col:
        show_table_key = session_key(key_prefix, "show_table")
        show_table = st.checkbox(
            "Показать таблицу",
            value=st.session_state.get(show_table_key, True),
            key=show_table_key,
        )
    return sort_mode, show_table


def prepare_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Подготовить датафрейм для AgGrid: сохранить сырые значения и форматированное отображение."""
    df_disp = df.copy()
    for tp, raw_col in RAW_COLUMNS.items():
        df_disp[raw_col] = df[tp]
        df_disp[tp] = [
            display_value_precise(value, mn, mx)
            for value, mn, mx in zip(df[tp], df[REF_MIN_COLUMN], df[REF_MAX_COLUMN])
        ]
    return df_disp


def build_grid_options(df_disp: pd.DataFrame, on_click: JsCode, color_cells: JsCode) -> dict:
    """Построить конфигурацию AgGrid."""
    builder = GridOptionsBuilder.from_dataframe(df_disp)
    builder.configure_grid_options(
        suppressMenu=True,
        suppressColumnMove=True,
        suppressSorting=True,
        suppressFilter=True,
        floatingFilter=False,
        rowSelection="single",
        rowDeselection=True,
        suppressRowClickSelection=True,
        onCellClicked=on_click,
        defaultColDef={"sortable": False, "resizable": True},
    )
    builder.configure_column(
        METABOLITE_COLUMN,
        width=AGGRID_COLUMN_WIDTHS[METABOLITE_COLUMN],
        cellStyle=_js_cell_style(METABOLITE_CELL_STYLE_JSON),
    )
    builder.configure_column(
        NORMS_COLUMN,
        width=AGGRID_COLUMN_WIDTHS[NORMS_COLUMN],
        cellStyle=_js_cell_style(NORMS_CELL_STYLE_JSON),
    )
    for tp in TIMEPOINTS:
        builder.configure_column(tp, width=TIMEPOINT_COLUMN_WIDTH, cellStyle=color_cells)
    for hidden in (REF_MIN_COLUMN, REF_MAX_COLUMN, *RAW_COLUMNS.values()):
        builder.configure_column(hidden, hide=True)
    return builder.build()


def _ensure_aggrid_css() -> None:
    """Добавить кастомные стили AgGrid один раз за сессию."""
    css_key = "__aggrid_css_applied"
    if st.session_state.get(css_key):
        return
    st.session_state[css_key] = True
    st.markdown(AGGRID_CUSTOM_CSS, unsafe_allow_html=True)


def render_grid_and_get_selection(
    df_disp: pd.DataFrame,
    grid_options: dict,
    key_prefix: str,
    show_table: bool,
    selected_key: str,
    current_selection: str | None,
) -> tuple[str | None, Any]:
    """Отрисовать таблицу и вернуть выбранный метаболит."""
    if show_table:
        _ensure_aggrid_css()
        table_col, graph_col = st.columns([9, 8], gap="medium")
        with table_col:
            grid_resp = AgGrid(
                df_disp,
                gridOptions=grid_options,
                height=AGGRID_DEFAULT_HEIGHT,
                allow_unsafe_jscode=True,
                theme=AGGRID_THEME,
                fit_columns_on_grid_load=True,
                update_on=["selectionChanged"],
                key=session_key("grid", key_prefix),
            )
            selected_rows = grid_resp.get("selected_rows", [])
            if isinstance(selected_rows, pd.DataFrame):
                selected_rows = selected_rows.to_dict(orient="records")
            if selected_rows:
                current_selection = selected_rows[0][METABOLITE_COLUMN]
                st.session_state[selected_key] = current_selection
        return current_selection, graph_col

    _, graph_col, _ = st.columns([1, 14, 1])
    return current_selection, graph_col


def _collect_comparison_lines(
    comparisons: list[dict[str, str]],
    sample: pd.DataFrame,
    ref_min: pd.Series,
    ref_max: pd.Series,
) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    base_tables: dict[str, pd.DataFrame] = {}
    for entry in comparisons:
        patient_code = entry.get("patient")
        metabolite = entry.get("metabolite")
        if not patient_code or not metabolite:
            continue
        if patient_code not in base_tables:
            base_tables[patient_code] = build_base_df(sample, patient_code, ref_min, ref_max)
        comp_df = base_tables[patient_code]
        comp_row = comp_df[comp_df[METABOLITE_COLUMN] == metabolite]
        if comp_row.empty:
            continue
        comp_record = comp_row.iloc[0]
        comp_values = [to_float(comp_record[tp]) for tp in TIMEPOINTS]
        comp_ref_min, comp_ref_max = reference_bounds(ref_min, ref_max, comp_record, metabolite)
        lines.append(
            {
                "patient": patient_code,
                "metabolite": metabolite,
                "values": comp_values,
                "ref_min": comp_ref_min,
                "ref_max": comp_ref_max,
            }
        )
    return lines


def _marker_colors(values: Iterable[Any], rmin: Any, rmax: Any) -> list[str]:
    lower = to_float(rmin)
    upper = to_float(rmax)

    def color(raw_value: Any) -> str:
        value = to_float(raw_value)
        if pd.isna(value) or pd.isna(lower) or pd.isna(upper):
            return COLOR_MISSING
        if value < lower - EPS:
            return COLOR_BELOW
        if value > upper + EPS:
            return COLOR_ABOVE
        return COLOR_NEUTRAL

    return [color(value) for value in values]


def _metabolite_chart_payload(
    df: pd.DataFrame,
    metabolite: str,
    ref_min: pd.Series,
    ref_max: pd.Series,
) -> tuple[list[float], float, float, list[str]] | None:
    row = df[df[METABOLITE_COLUMN] == metabolite]
    if row.empty:
        return None
    record = row.iloc[0]
    values = [to_float(record[tp]) for tp in TIMEPOINTS]
    rmin, rmax = reference_bounds(ref_min, ref_max, record, metabolite)
    colors = _marker_colors(values, rmin, rmax)
    return values, rmin, rmax, colors


def _collect_numeric_values(
    base_values: Iterable[float],
    comparison_lines: list[dict[str, Any]],
) -> list[float]:
    collected = [value for value in base_values if pd.notna(value)]
    for line in comparison_lines:
        collected.extend(value for value in line["values"] if pd.notna(value))
    return collected


def _chart_axis_limits(rmin: float, rmax: float, values: list[float]) -> tuple[float, float]:
    if values:
        lower_bound = min([rmin, *values])
        upper_bound = max([rmax, *values])
    else:
        lower_bound, upper_bound = rmin, rmax

    if upper_bound > lower_bound:
        pad = (upper_bound - lower_bound) * DEFAULT_PADDING_RATIO
    else:
        pad = (abs(upper_bound) + 1.0) * DEFAULT_PADDING_RATIO

    y0 = max(0.0, lower_bound - pad)
    y1 = upper_bound + pad
    return y0, y1


def _reference_bands(y0: float, rmin: float, rmax: float, y1: float) -> list[dict[str, Any]]:
    return [
        {"y0": y0, "y1": rmin, "fillcolor": "rgba(127,174,230,0.35)", "line_width": 0},
        {"y0": rmin, "y1": rmax, "fillcolor": "rgba(200,200,200,0.35)", "line_width": 0},
        {"y0": rmax, "y1": y1, "fillcolor": "rgba(230,115,115,0.35)", "line_width": 0},
    ]


def render_comparison_chart(
    df: pd.DataFrame,
    metabolite: str,
    base_code: str,
    comparisons: list[dict[str, str]],
    sample: pd.DataFrame,
    ref_min: pd.Series,
    ref_max: pd.Series,
) -> None:
    """Отрисовать график сравнения значений метаболита по таймпоинтам."""
    payload = _metabolite_chart_payload(df, metabolite, ref_min, ref_max)
    if payload is None:
        st.info("Выберите метаболит в таблице.")
        return
    base_values, rmin, rmax, marker_colors = payload

    comparison_lines = _collect_comparison_lines(comparisons, sample, ref_min, ref_max)
    numeric_values = _collect_numeric_values(base_values, comparison_lines)
    y0, y1 = _chart_axis_limits(rmin, rmax, numeric_values)

    fig = go.Figure()
    for band in _reference_bands(y0, rmin, rmax, y1):
        fig.add_hrect(**band)

    fig.add_trace(
        go.Scatter(
            x=list(TIMEPOINTS),
            y=base_values,
            mode="lines+markers",
            name=f"{base_code} · {metabolite}",
            line=dict(color=BASE_LINE_COLOR, dash="dot"),
            hoverinfo="x+y+name",
            marker=dict(
                color=marker_colors,
                size=10,
                symbol="circle",
                line=dict(width=1, color="rgba(0,0,0,0.35)"),
            ),
        )
    )

    palette = plotly_qual.Plotly
    for idx, line in enumerate(comparison_lines):
        color = palette[idx % len(palette)]
        comp_marker_colors = _marker_colors(line["values"], rmin, rmax)
        fig.add_trace(
            go.Scatter(
                x=list(TIMEPOINTS),
                y=line["values"],
                mode="lines+markers",
                name=f"{line['patient']} · {line['metabolite']}",
                line=dict(color=color, dash="dash"),
                hoverinfo="x+y+name",
                marker=dict(
                    color=comp_marker_colors,
                    size=9,
                    symbol="circle",
                    line=dict(width=1, color="rgba(0,0,0,0.35)"),
                ),
            )
        )

    fig.update_layout(
        title=dict(text=metabolite, y=0.96, yanchor="top"),
        template="simple_white",
        height=DEFAULT_FIG_HEIGHT,
        margin=dict(l=10, r=170, t=70, b=60),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#d9d9d9",
            borderwidth=1,
            font=dict(size=11, color="#212121"),
            itemclick="toggleothers",
            itemdoubleclick="toggle",
        ),
        xaxis_title="Точка измерения",
        yaxis_title="Концентрация, нмоль",
    )
    fig.update_xaxes(categoryorder="array", categoryarray=list(TIMEPOINTS))
    fig.update_yaxes(range=[y0, y1], ticks="outside", showgrid=True)
    st.plotly_chart(fig, use_container_width=True)


def _render_comparison_controls(
    key_prefix: str,
    clicked: str,
    patient_codes: list[str],
    sample: pd.DataFrame,
    ref_min: pd.Series,
    ref_max: pd.Series,
) -> list[dict[str, str]]:
    compare_key = session_key("compare", key_prefix)
    entries = list(st.session_state.get(compare_key, []))

    if st.button(COMPARISON_BUTTON_LABEL, key=session_key("add_compare", key_prefix)):
        entries.append({"patient": "", "metabolite": clicked})

    updated: list[dict[str, str]] = []
    for idx, entry in enumerate(entries):
        cols = st.columns([4, 4, 1])
        patient_options = [PLACEHOLDER, *patient_codes]
        current_patient = entry.get("patient") or PLACEHOLDER
        selected_patient = cols[0].selectbox(
            "Пациент",
            patient_options,
            index=patient_options.index(current_patient) if current_patient in patient_options else 0,
            key=session_key("compare_patient", key_prefix, idx),
        )
        entry["patient"] = "" if selected_patient == PLACEHOLDER else selected_patient

        metabolite_options = [PLACEHOLDER]
        if entry["patient"]:
            comp_df = build_base_df(sample, entry["patient"], ref_min, ref_max)
            metabolite_options += comp_df[METABOLITE_COLUMN].tolist()

        current_met = entry.get("metabolite") or clicked
        selected_metabolite = cols[1].selectbox(
            "Метаболит",
            metabolite_options,
            index=metabolite_options.index(current_met) if current_met in metabolite_options else 0,
            key=session_key("compare_metabolite", key_prefix, idx),
        )
        entry["metabolite"] = "" if selected_metabolite == PLACEHOLDER else selected_metabolite

        remove = cols[2].button("✖", key=session_key("remove_compare", key_prefix, idx))
        if not remove:
            updated.append(entry)

    st.session_state[compare_key] = updated
    return [entry for entry in updated if entry.get("patient") and entry.get("metabolite")]


def _ensure_selection(sorted_df: pd.DataFrame, selected_key: str, current_selection: str | None) -> str | None:
    available = set(sorted_df[METABOLITE_COLUMN])
    if current_selection in available:
        return current_selection
    if sorted_df.empty:
        return None
    first_value = sorted_df.iloc[0][METABOLITE_COLUMN]
    st.session_state[selected_key] = first_value
    return first_value


def _prepare_table_data(df: pd.DataFrame, sort_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_df = sort_by_status(df, sort_mode)
    display_df = prepare_display_dataframe(sorted_df)
    return sorted_df, display_df


def _render_chart_panel(
    sorted_df: pd.DataFrame,
    clicked: str | None,
    key_prefix: str,
    base_code: str,
    patient_codes: list[str],
    sample: pd.DataFrame,
    ref_min: pd.Series,
    ref_max: pd.Series,
) -> None:
    if not clicked:
        st.info("Выберите метаболит в таблице слева.")
        return

    comparisons = _render_comparison_controls(
        key_prefix=key_prefix,
        clicked=clicked,
        patient_codes=patient_codes,
        sample=sample,
        ref_min=ref_min,
        ref_max=ref_max,
    )

    render_comparison_chart(
        df=sorted_df,
        metabolite=clicked,
        base_code=base_code,
        comparisons=comparisons,
        sample=sample,
        ref_min=ref_min,
        ref_max=ref_max,
    )


def render_table_and_plot(
    df: pd.DataFrame,
    key_prefix: str,
    base_code: str,
    sample: pd.DataFrame,
    ref_min: pd.Series,
    ref_max: pd.Series,
    patient_codes: list[str],
) -> None:
    """Главный блок отрисовки таблицы и графика."""
    if df.empty:
        st.warning("Нет данных для отображения.")
        return

    selected_key = session_key("selected_metabolite", key_prefix)
    clicked = st.session_state.get(selected_key)

    sort_mode, show_table = render_table_controls(key_prefix)
    sorted_df, df_disp = _prepare_table_data(df, sort_mode)

    on_click = JsCode(AGGRID_ON_CELL_CLICK_JS)
    color_cells = _color_cells_js()
    grid_options = build_grid_options(df_disp, on_click, color_cells)

    clicked, graph_col = render_grid_and_get_selection(
        df_disp=df_disp,
        grid_options=grid_options,
        key_prefix=key_prefix,
        show_table=show_table,
        selected_key=selected_key,
        current_selection=clicked,
    )
    clicked = _ensure_selection(sorted_df, selected_key, clicked)

    with graph_col:
        _render_chart_panel(
            sorted_df=sorted_df,
            clicked=clicked,
            key_prefix=key_prefix,
            base_code=base_code,
            patient_codes=patient_codes,
            sample=sample,
            ref_min=ref_min,
            ref_max=ref_max,
        )


# The basic logic
def main() -> None:
    st.set_page_config(page_title="Итоговая таблица пациентов", layout="wide")
    st.title("Итоговая таблица метаболитов")

    try:
        ref_min, ref_max, risk_map = load_reference_data(REF_PATH)
    except FileNotFoundError:
        st.error("Файл Ref.xlsx не найден рядом с приложением.")
        st.stop()
    except ValueError as exc:
        st.error(f"Ref.xlsx имеет некорректный формат: {exc}")
        st.stop()

    st.sidebar.header("Загрузка данных")
    uploaded_sample = st.sidebar.file_uploader("Загрузите файл с данными (.xlsx)", type=["xlsx"])
    if uploaded_sample is None:
        st.sidebar.warning("Загрузите Excel с данными пациентов.")
        st.stop()

    try:
        sample = load_sample(uploaded_sample)
    except ValueError as exc:
        st.sidebar.error(str(exc))
        st.stop()
    else:
        st.sidebar.success("Файл данных успешно загружен.")

    base_codes_series = (
        sample[BASE_CODE_COLUMN]
        if BASE_CODE_COLUMN in sample.columns
        else sample["Код"].apply(_base_patient_code)
    )
    patient_codes = sorted({code for code in (str(item).strip() for item in base_codes_series) if code})
    if not patient_codes:
        st.warning("В выгрузке не найдено пациентов.")
        st.stop()

    st.markdown("### Режим")
    mode = st.radio(
        "Выберите режим:",
        ["Все метаболиты", "Риски"],
        horizontal=True,
        label_visibility="collapsed",
        key="global_mode",
    )

    st.sidebar.header("Пациенты")
    patient_key = session_key("selected_patient")
    default_patient = st.session_state.get(patient_key, patient_codes[0])
    if default_patient not in patient_codes:
        default_patient = patient_codes[0]
    selected_code = st.sidebar.radio(
        "Выберите пациента:",
        patient_codes,
        index=patient_codes.index(default_patient),
    )
    st.session_state[patient_key] = selected_code

    st.markdown("---")
    st.subheader(f"Пациент: {selected_code}")

    base_df = build_base_df(sample, selected_code, ref_min, ref_max)

    if mode == "Все метаболиты":
        render_table_and_plot(
            df=base_df,
            key_prefix=f"{selected_code}_all",
            base_code=selected_code,
            sample=sample,
            ref_min=ref_min,
            ref_max=ref_max,
            patient_codes=patient_codes,
        )
        return

    if not risk_map:
        st.warning("В Ref.xlsx не найдена корректная таблица рисков (лист 2).")
        return

    risk_names = sorted(risk_map.keys())
    risk_key = session_key("risk_choice", selected_code)
    previous_risk = st.session_state.get(risk_key, risk_names[0] if risk_names else "")
    risk_choice = st.selectbox(
        "Выберите группу риска:",
        risk_names,
        index=risk_names.index(previous_risk) if previous_risk in risk_names else 0,
        key=risk_key,
    )

    metabolites_in_risk = set(risk_map.get(risk_choice, []))
    risk_df = base_df[base_df[METABOLITE_COLUMN].isin(metabolites_in_risk)].copy()

    st.markdown(f"#### {risk_choice}")
    if risk_df.empty:
        st.info("Для выбранного риска нет метаболитов в данных этого пациента.")
    else:
        render_table_and_plot(
            df=risk_df,
            key_prefix=f"{selected_code}_risk",
            base_code=selected_code,
            sample=sample,
            ref_min=ref_min,
            ref_max=ref_max,
            patient_codes=patient_codes,
        )


if __name__ == "__main__":
    main()