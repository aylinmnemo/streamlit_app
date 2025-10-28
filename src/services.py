from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    BASE_CODE_COLUMN,
    COLOR_ABOVE,
    COLOR_BELOW,
    COLOR_BG_ABOVE,
    COLOR_BG_BELOW,
    COLOR_BG_MISSING,
    COLOR_BG_NORMAL,
    COLOR_MISSING,
    COLOR_NEUTRAL,
    DEFAULT_PADDING_RATIO,
    EPS,
    METABOLITE_COLUMN,
    NORMS_COLUMN,
    PATIENT_CACHE_TTL,
    PLACEHOLDER,
    RAW_COLUMNS,
    REF_MAX_COLUMN,
    REF_MIN_COLUMN,
    REQUIRED_SAMPLE_COLUMNS,
    SortMode,
    TIMEPOINTS,
)
from src.data import (
    display_value_precise,
    reference_bounds,
    resolve_tp,
    to_float,
)


@dataclass(frozen=True)
class MetaboliteRange:
    name: str
    mn: float
    mx: float

    def norm_text(self) -> str:
        return f"{self.mn:g} - {self.mx:g}"


@dataclass
class PatientSnapshot:
    code: str
    rows: pd.DataFrame
    ranges: dict[str, MetaboliteRange]

    @property
    def metabolites(self) -> list[str]:
        return [metabolite for metabolite in self.ranges if metabolite in self.rows.columns]

    def table(self) -> pd.DataFrame:
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


def build_patient_table(sample_df: pd.DataFrame, base_code: str, ref_min: pd.Series, ref_max: pd.Series) -> pd.DataFrame:
    if BASE_CODE_COLUMN in sample_df.columns:
        rows = sample_df[sample_df[BASE_CODE_COLUMN] == base_code].copy()
    else:
        rows = sample_df[sample_df["Код"].astype(str).str.contains(fr"\b{base_code}\b", case=False, na=False)].copy()
    excluded_columns = set(REQUIRED_SAMPLE_COLUMNS) | {BASE_CODE_COLUMN}
    available_metabolites = [column for column in rows.columns if column not in excluded_columns]
    selected = [name for name in available_metabolites if name in ref_min.index and name in ref_max.index]
    filtered_min = ref_min.reindex(selected)
    filtered_max = ref_max.reindex(selected)
    ranges = {
        name: MetaboliteRange(name, filtered_min[name], filtered_max[name])
        for name in selected
    }
    snapshot = PatientSnapshot(base_code, rows, ranges)
    return snapshot.table()


@st.cache_data(show_spinner=False, ttl=PATIENT_CACHE_TTL)
def cached_patient_table(
    base_code: str,
    sample_df: pd.DataFrame,
    ref_min: pd.Series,
    ref_max: pd.Series,
) -> pd.DataFrame:
    return build_patient_table(sample_df, base_code, ref_min, ref_max)


def add_deviation_counts(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    values = enriched.loc[:, TIMEPOINTS]
    below_threshold = enriched[REF_MIN_COLUMN] - EPS
    above_threshold = enriched[REF_MAX_COLUMN] + EPS
    enriched["_below_cnt"] = values.lt(below_threshold, axis=0).sum(axis=1)
    enriched["_above_cnt"] = values.gt(above_threshold, axis=0).sum(axis=1)
    return enriched


def sort_by_status(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    enriched = add_deviation_counts(df)
    if mode == SortMode.ALPHA.value:
        return enriched.sort_values(METABOLITE_COLUMN, ascending=True, ignore_index=True).drop(
            columns=["_below_cnt", "_above_cnt"]
        )

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


def prepare_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_disp = df.copy()
    for tp, raw_col in RAW_COLUMNS.items():
        df_disp[raw_col] = df[tp]
        df_disp[tp] = [
            display_value_precise(value, mn, mx)
            for value, mn, mx in zip(df[tp], df[REF_MIN_COLUMN], df[REF_MAX_COLUMN])
        ]
    return df_disp


def collect_comparison_lines(
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
            base_tables[patient_code] = cached_patient_table(patient_code, sample, ref_min, ref_max)
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


def patient_metabolite_line(
    patient_code: str,
    metabolite: str,
    sample: pd.DataFrame,
    ref_min: pd.Series,
    ref_max: pd.Series,
) -> dict[str, Any] | None:
    patient_df = cached_patient_table(patient_code, sample, ref_min, ref_max)
    row = patient_df[patient_df[METABOLITE_COLUMN] == metabolite]
    if row.empty:
        return None
    record = row.iloc[0]
    values = [to_float(record[tp]) for tp in TIMEPOINTS]
    if all(pd.isna(value) for value in values):
        return None
    rmin, rmax = reference_bounds(ref_min, ref_max, record, metabolite)
    return {
        "patient": patient_code,
        "values": values,
        "rmin": rmin,
        "rmax": rmax,
        "colors": marker_colors(values, rmin, rmax),
    }


def marker_colors(values: Iterable[Any], rmin: Any, rmax: Any) -> list[str]:
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


def metabolite_chart_payload(
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
    colors = marker_colors(values, rmin, rmax)
    return values, rmin, rmax, colors


def collect_numeric_values(
    base_values: Iterable[float],
    comparison_lines: list[dict[str, Any]],
) -> list[float]:
    collected = [value for value in base_values if pd.notna(value)]
    for line in comparison_lines:
        collected.extend(value for value in line["values"] if pd.notna(value))
    return collected


def chart_axis_limits(rmin: float, rmax: float, values: list[float]) -> tuple[float, float]:
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


def reference_bands(y0: float, rmin: float, rmax: float, y1: float) -> list[dict[str, Any]]:
    return [
        {"y0": y0, "y1": rmin, "fillcolor": "rgba(127,174,230,0.35)", "line_width": 0},
        {"y0": rmin, "y1": rmax, "fillcolor": "rgba(200,200,200,0.35)", "line_width": 0},
        {"y0": rmax, "y1": y1, "fillcolor": "rgba(230,115,115,0.35)", "line_width": 0},
    ]
