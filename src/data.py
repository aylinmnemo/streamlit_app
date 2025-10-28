from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st

from src.config import (
    BASE_CODE_COLUMN,
    BASE_CODE_SUFFIX_PATTERN,
    CODE_SUFFIX_PATTERN,
    EPS,
    GROUP_KEYWORDS,
    REF_MAX_COLUMN,
    REF_MIN_COLUMN,
    REQUIRED_SAMPLE_COLUMNS,
    SUFFIX_TO_TIMEPOINT,
    TimePoint,
)


def to_float(value: Any) -> float:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return float("nan")
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return float("nan")


def session_key(*parts: Iterable[Any]) -> str:
    return "__".join(str(part) for part in parts if part is not None and part != "")


def resolve_tp(code: str, group: str) -> str:
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
    rmin = to_float(record.get(REF_MIN_COLUMN, float("nan")))
    rmax = to_float(record.get(REF_MAX_COLUMN, float("nan")))
    if pd.isna(rmin):
        rmin = lookup_reference(ref_min_series, metabolite)
    if pd.isna(rmax):
        rmax = lookup_reference(ref_max_series, metabolite)
    return rmin, rmax


@st.cache_data(show_spinner=False, ttl=300)
def load_reference_data(path: Path) -> tuple[pd.Series, pd.Series, dict[str, list[str]]]:
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


def _base_patient_code(code: Any) -> str:
    text = str(code).strip()
    if not text or text.lower() == "nan":
        return ""
    suffix_match = CODE_SUFFIX_PATTERN.search(text)
    if suffix_match:
        suffix_digits = suffix_match.group(1)
        if suffix_digits in SUFFIX_TO_TIMEPOINT:
            base = text[: suffix_match.start()].rstrip("-").strip()
            if base:
                return base
    if text.count("-") >= 2:
        return BASE_CODE_SUFFIX_PATTERN.sub("", text)
    return text


def load_sample(uploaded_file: Any) -> pd.DataFrame:
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
        df = df.assign(**converted).copy()

    df["Код"] = df["Код"].astype(str).str.strip()
    df[BASE_CODE_COLUMN] = df["Код"].apply(_base_patient_code)
    return df


def get_patient_codes(sample_df: pd.DataFrame) -> list[str]:
    source = (
        sample_df[BASE_CODE_COLUMN]
        if BASE_CODE_COLUMN in sample_df.columns
        else sample_df["Код"].apply(_base_patient_code)
    )
    return sorted({code for code in (str(item).strip() for item in source) if code})
