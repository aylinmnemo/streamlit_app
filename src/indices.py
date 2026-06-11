import re
import pandas as pd
import numpy as np

EPS = 1e-9
TIMEPOINTS = ["До", "После", "След.день"]
DEMO_MIN_REFERENCE_N = 5

FINAL_WEIGHTS = {
    "ERI": 0.35,
    "AARI": 0.25,
    "NRI": 0.25,
    "CRI": 0.15,
}

INDEX_WEIGHTS = {
    "ERI": {
        "C2/C0": 1.4,
        "Соотношение ацетилкарнитина к карнитину": 1.4,
        "(C2+C3)/C0": 1.3,
        "Sum of ACs/C0": 1.3,
        "C8": 1.2,
        "C10": 1.2,
        "C12-1": 1.1,
        "C14-2": 1.1,
        "(C16+C18)/C2": 1.1,
        "C8-1": 1.0,
        "ССК/СДК": 1.0,
        "СКК/СДК": 1.0,
        "СКК/ССК": 0.9,
        "(C6+C8+C10)/C2": 1.0,
    },
    "AARI": {
        "Alanine": 1.5,
        "Valine/Alanine": 1.4,
        "Asparagine": 1.2,
        "Arg/Orn+Cit": 1.2,
        "Ala/Gly": 1.1,
        "Alanine/Valine": 1.0,
    },
    "NRI": {
        "Quin/HIAA": 1.5,
        "HIAA": 1.4,
        "5-hydroxytryptophan": 1.3,
        "Kynurenic acid/Kynurenine": 1.3,
        "Kynurenic acid": 1.2,
        "GABR": 1.0,
    },
    "CRI": {
        "Pantothenic": 1.5,
        "Riboflavin/Pantothenic": 1.3,
        "Uridine": 1.2,
    },
}

ALL_INDEX_MARKERS = sorted({
    marker
    for group in INDEX_WEIGHTS.values()
    for marker in group.keys()
})


def _norm_name(value: object) -> str:
    """Нормализация названий колонок для устойчивого сопоставления."""
    text = str(value).strip().lower().replace("ё", "е")
    text = text.replace("с", "c")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace(":", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*\+\s*", "+", text)
    text = re.sub(r"\s*\(\s*", "(", text)
    text = re.sub(r"\s*\)\s*", ")", text)
    return text.strip()


COLUMN_ALIASES = {
    "C2/C0": ["C2/C0", "С2/С0", "C2 / C0", "С2 / С0"],
    "Соотношение ацетилкарнитина к карнитину": [
        "Соотношение ацетилкарнитина к карнитину",
        "Ratio of Acetylcarnitine to Carnitine",
    ],
    "(C2+C3)/C0": ["(C2+C3)/C0", "(C2 + C3) / C0", "(С2+С3)/С0"],
    "Sum of ACs/C0": ["Sum of ACs/C0", "Сумма ацилкарнитинов/С0", "Сумма ацилкарнитинов / С0"],
    "C8": ["C8", "С8"],
    "C10": ["C10", "С10"],
    "C12-1": ["C12-1", "C12:1", "С12-1", "С12:1"],
    "C14-2": ["C14-2", "C14:2", "С14-2", "С14:2"],
    "(C16+C18)/C2": ["(C16+C18)/C2", "(C16 + C18) / C2", "(С16+С18)/С2"],
    "C8-1": ["C8-1", "C8:1", "С8-1", "С8:1"],
    "ССК/СДК": ["ССК/СДК", "Ratio of Medium-Chain to Long-Chain ACs"],
    "СКК/СДК": ["СКК/СДК", "Ratio of Short-Chain to Long-Chain ACs"],
    "СКК/ССК": ["СКК/ССК", "Ratio of Short-Chain to Medium-Chain ACs"],
    "(C6+C8+C10)/C2": ["(C6+C8+C10)/C2", "(C6 + C8 + C10) / C2", "(С6+С8+С10)/С2"],
    "Alanine": ["Alanine", "Аланин"],
    "Valine/Alanine": ["Valine/Alanine", "Valine / Alanine", "Валин/Аланин", "Валин / Аланин"],
    "Asparagine": ["Asparagine", "Аспарагин"],
    "Arg/Orn+Cit": [
        "Arg/Orn+Cit",
        "Аргинин/Орнитин+Цитруллин",
        "Аргинин/(Орнитин+Цитруллин)",
    ],
    "Ala/Gly": ["Ala/Gly", "Аланин/Глицин", "Аланин / Глицин"],
    "Alanine/Valine": ["Alanine/Valine", "Alanine / Valine", "Аланин/Валин", "Аланин / Валин"],
    "Quin/HIAA": [
        "Quin/HIAA",
        "Хинолиновая кислота/5-Гидроксииндолуксусная кислота",
    ],
    "HIAA": ["HIAA", "5-HIAA", "5-Гидроксииндолуксусная кислота"],
    "5-hydroxytryptophan": ["5-hydroxytryptophan", "5-гидрокситриптофан", "5-Гидрокситриптофан"],
    "Kynurenic acid/Kynurenine": [
        "Kynurenic acid/Kynurenine",
        "Kynurenic acid / Kynurenine",
        "Кинуреновая кислота/Кинуренин",
    ],
    "Kynurenic acid": ["Kynurenic acid", "Кинуреновая кислота"],
    "GABR": ["GABR", "GABR (Аргинин/(Орнитин+Цитруллин))"],
    "Pantothenic": ["Pantothenic", "Пантотеновая кислота"],
    "Riboflavin/Pantothenic": [
        "Riboflavin/Pantothenic",
        "Riboflavin / Pantothenic",
        "Рибофлавин/Пантотеновая кислота",
        "Рибофлавин / Пантотеновая кислота",
    ],
    "Uridine": ["Uridine", "Уридин"],
}

ALIAS_TO_CANONICAL = {
    _norm_name(alias): canonical
    for canonical, aliases in COLUMN_ALIASES.items()
    for alias in aliases
}


def extract_patient(code):
    text = str(code).strip()
    if not text or text.lower() == "nan":
        return ""
    text = re.sub(r"[-–—_]\s*[ТT]\s*[0-2]\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[-–—_]\s*[1-3]\s*$", "", text, flags=re.IGNORECASE)
    return text.strip(" -–—_")


def extract_timepoint(code, group=""):
    group_text = str(group).strip().lower().replace("ё", "е")
    code_text = str(code).strip().lower().replace("т", "t")

    if any(word in group_text for word in ["след", "24", "recovery", "next", "сут"]):
        return "След.день"
    if any(word in group_text for word in ["после", "after", "post"]):
        return "После"
    if any(word in group_text for word in ["до", "baseline", "before", "pre"]):
        return "До"
    if re.search(r"[-–—_]\s*t0\s*$", code_text) or re.search(r"[-–—_]\s*1\s*$", code_text):
        return "До"
    if re.search(r"[-–—_]\s*t1\s*$", code_text) or re.search(r"[-–—_]\s*2\s*$", code_text):
        return "После"
    if re.search(r"[-–—_]\s*t2\s*$", code_text) or re.search(r"[-–—_]\s*3\s*$", code_text):
        return "След.день"
    return ""


def find_column(df, possible_names):
    lower_columns = {str(column).strip().lower(): column for column in df.columns}
    for name in possible_names:
        key = name.strip().lower()
        if key in lower_columns:
            return lower_columns[key]
    return None


def _rename_index_columns(df):
    """Переименовывает известные маркеры в канонические названия."""
    rename_map = {}
    used = set()
    for col in df.columns:
        canonical = ALIAS_TO_CANONICAL.get(_norm_name(col))
        if canonical is not None and canonical not in used:
            rename_map[col] = canonical
            used.add(canonical)
    return df.rename(columns=rename_map)


def prepare_index_data(df):
    df = _rename_index_columns(df.copy())

    patient_col = find_column(df, ["Пациент", "Patient", "ID", "patient_id", "Участник"])
    timepoint_col = find_column(df, ["Точка", "Группа", "Timepoint", "Time point", "Group", "Visit"])
    code_col = find_column(df, ["Код", "Code", "Sample", "Sample ID", "Sample_ID"])

    if code_col is not None:
        df["Пациент"] = df[code_col].apply(extract_patient)
        df["Точка"] = df.apply(
            lambda row: extract_timepoint(row[code_col], row[timepoint_col]) if timepoint_col is not None else extract_timepoint(row[code_col], ""),
            axis=1,
        )
    elif patient_col is not None:
        df["Пациент"] = df[patient_col].apply(extract_patient)
        df["Точка"] = df.apply(
            lambda row: extract_timepoint(row[patient_col], row[timepoint_col]) if timepoint_col is not None else extract_timepoint(row[patient_col], ""),
            axis=1,
        )
    else:
        raise ValueError("Не найдена колонка пациента/кода образца: нужна 'Код', 'Пациент', 'ID' или аналогичная.")

    df = df[df["Пациент"].astype(str).str.len() > 0]
    df = df[df["Точка"].isin(TIMEPOINTS)]
    if df.empty:
        raise ValueError("Не удалось распознать временные точки: используй Т0/Т1/Т2, 1/2/3 или До/После/След.день.")

    marker_columns = [marker for marker in ALL_INDEX_MARKERS if marker in df.columns]
    if not marker_columns:
        raise ValueError("Не найдены колонки маркеров для расчета индексов. Проверь названия метаболитов/соотношений.")

    for column in marker_columns:
        df[column] = pd.to_numeric(df[column].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        # log2 допустим только для положительных 
        df.loc[df[column] <= 0, column] = np.nan
        df[column] = np.log2(df[column] + EPS)

    return df, marker_columns


def make_pivot(df):
    prepared_df, marker_columns = prepare_index_data(df)
    pivot_df = prepared_df.pivot_table(
        index="Пациент",
        columns="Точка",
        values=marker_columns,
        aggfunc="last",
    )
    return pivot_df, marker_columns

def _calculate_demo_recovery_scores_from_pivot(pivot_df, marker_columns):
    """
    DEMO fallback для маленьких файлов.
    """
    recovery_scores = pd.DataFrame(index=pivot_df.index)

    for marker in marker_columns:
        needed = [(marker, "До"), (marker, "После"), (marker, "След.день")]
        if not all(col in pivot_df.columns for col in needed):
            continue

        x0 = pivot_df[(marker, "До")]
        x1 = pivot_df[(marker, "После")]
        x2 = pivot_df[(marker, "След.день")]

        denominator = (x1 - x0).replace(0, np.nan)

        score = 1 - np.abs((x2 - x0) / denominator)
        score = score.replace([np.inf, -np.inf], np.nan)

        recovery_scores[marker] = score.clip(0, 1) * 10

    return recovery_scores.round(2)

def calculate_delta_orp_scores(df):
    """ΔScoreᵢ = 10 * min(|x1 - x0| / Δref, 1)."""
    pivot_df, marker_columns = make_pivot(df)
    delta_raw = {}

    for marker in marker_columns:
        needed = [(marker, "До"), (marker, "После")]
        if all(col in pivot_df.columns for col in needed):
            delta_raw[marker] = (pivot_df[(marker, "После")] - pivot_df[(marker, "До")]).abs()

    delta_raw = pd.DataFrame(delta_raw, index=pivot_df.index)
    delta_scores = pd.DataFrame(index=delta_raw.index)

    for marker in delta_raw.columns:
        values = delta_raw[marker].dropna()
        delta_ref = np.nanpercentile(values, 90) if len(values) else np.nan
        if pd.isna(delta_ref) or delta_ref <= 0:
            delta_scores[marker] = np.nan
        else:
            delta_scores[marker] = 10 * np.minimum(delta_raw[marker] / delta_ref, 1)

    return delta_scores.round(2)


def calculate_orp_scores(df):
    pivot_df, marker_columns = make_pivot(df)

    if len(pivot_df.index) < DEMO_MIN_REFERENCE_N:
        return _calculate_demo_recovery_scores_from_pivot(pivot_df, marker_columns)

    htsi_scores = pd.DataFrame(index=pivot_df.index)

    for marker in marker_columns:
        needed = [(marker, "До"), (marker, "После"), (marker, "След.день")]
        if not all(col in pivot_df.columns for col in needed):
            continue

        z_values = []
        for tp in TIMEPOINTS:
            x = pivot_df[(marker, tp)]
            mu = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - mu))
            denom = 1.4826 * mad
            if pd.isna(denom) or denom <= 0:
                # Защита от деления на ноль 
                denom = np.nanstd(x)
            if pd.isna(denom) or denom <= 0:
                denom = EPS
            z_values.append((x - mu) / denom)

        d = np.sqrt((z_values[0] ** 2 + z_values[1] ** 2 + z_values[2] ** 2) / 3)
        valid_d = d.dropna()
        d_ref = np.nanpercentile(valid_d, 90) if len(valid_d) else np.nan

        if pd.isna(d_ref) or d_ref <= 0:
            htsi_scores[marker] = np.nan
        else:
            htsi_scores[marker] = 10 * np.exp(-0.6 * (d / d_ref) ** 2)
            htsi_scores[marker] = htsi_scores[marker].clip(0, 10)

    return htsi_scores.round(2)


def weighted_mean(row, marker_weights):
    values, weights = [], []
    for marker, weight in marker_weights.items():
        if marker in row.index and pd.notna(row[marker]):
            values.append(row[marker])
            weights.append(weight)
    if not values:
        return np.nan
    return np.average(values, weights=weights)


def interpret_orp(value):
    if pd.isna(value):
        return "Недостаточно данных"
    if value >= 8:
        return "Высокая близость к физиологическому восстановлению"
    if value >= 6:
        return "Сохраненное восстановление с умеренными отклонениями"
    if value >= 4:
        return "Напряженное или замедленное восстановление"
    return "Выраженное отклонение от физиологического восстановления"


def interpret_delta_orp(value):
    if pd.isna(value):
        return "Недостаточно данных"
    if value >= 8:
        return "Выраженный динамический отклик"
    if value >= 6:
        return "Умеренно выраженный динамический отклик"
    if value >= 4:
        return "Слабый или умеренный динамический отклик"
    return "Минимальный динамический отклик"


def _summary_from_scores(scores, suffix=""):
    result = pd.DataFrame(index=scores.index)
    for index_name, marker_weights in INDEX_WEIGHTS.items():
        result[f"{index_name}{suffix}"] = scores.apply(lambda row: weighted_mean(row, marker_weights), axis=1)

    result[f"ORP{suffix}" if not suffix else f"ΔORP"] = (
        FINAL_WEIGHTS["ERI"] * result[f"ERI{suffix}"]
        + FINAL_WEIGHTS["AARI"] * result[f"AARI{suffix}"]
        + FINAL_WEIGHTS["NRI"] * result[f"NRI{suffix}"]
        + FINAL_WEIGHTS["CRI"] * result[f"CRI{suffix}"]
    )
    return result


def calculate_orp_summary(df):
    orp_scores = calculate_orp_scores(df)
    result = _summary_from_scores(orp_scores, suffix="")
    result["Интерпретация ORP"] = result["ORP"].apply(interpret_orp)
    return result.round(2).reset_index()


def calculate_delta_orp_summary(df):
    delta_scores = calculate_delta_orp_scores(df)
    result = _summary_from_scores(delta_scores, suffix="_Δ")
    result["Интерпретация ΔORP"] = result["ΔORP"].apply(interpret_delta_orp)
    return result.round(2).reset_index()


def calculate_full_index_summary(df):
    orp_summary = calculate_orp_summary(df)
    delta_summary = calculate_delta_orp_summary(df)
    return orp_summary.merge(delta_summary, on="Пациент", how="outer")

def get_index_calculation_mode(df):
    """
    Возвращает режим расчета индексов:
    demo — если участников мало и ORP считается по recovery fallback;
    reference — если участников достаточно для median/MAD ORP.
    """
    pivot_df, _ = make_pivot(df)
    n_patients = len(pivot_df.index)

    if n_patients < DEMO_MIN_REFERENCE_N:
        return "demo", n_patients

    return "reference", n_patients