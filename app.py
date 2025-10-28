import streamlit as st

from src.config import METABOLITE_COLUMN, REF_PATH
from src.data import get_patient_codes, load_reference_data, load_sample, session_key
from src.services import cached_patient_table
from src.ui import render_multi_patient_mode, render_risk_overview_chart, render_table_and_plot


def _render_patient_sidebar(patient_codes: list[str]) -> list[str]:
    """Показать чекбоксы пациентов и вернуть выбранные коды."""
    st.sidebar.header("Пациенты")
    selection_key = session_key("patients", "selected")
    default_selection = st.session_state.get(selection_key, [patient_codes[0]])
    default_selection = [code for code in default_selection if code in patient_codes] or [patient_codes[0]]

    container = st.sidebar.container()
    container.markdown("Отметьте пациентов для анализа:")

    selected: list[str] = []
    for code in patient_codes:
        checkbox_key = session_key("patients", "choice", code)
        if checkbox_key not in st.session_state:
            st.session_state[checkbox_key] = code in default_selection
        if container.checkbox(code, key=checkbox_key):
            selected.append(code)

    st.session_state[selection_key] = selected
    return selected


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

    patient_codes = get_patient_codes(sample)
    if not patient_codes:
        st.warning("В выгрузке не найдено пациентов.")
        st.stop()

    selected_patients = _render_patient_sidebar(patient_codes)
    if not selected_patients:
        st.sidebar.warning("Отметьте хотя бы одного пациента для просмотра данных.")
        st.stop()

    if len(selected_patients) >= 2:
        st.markdown("### Сравнение пациентов")
        render_multi_patient_mode(
            selected_patients=selected_patients,
            sample=sample,
            ref_min=ref_min,
            ref_max=ref_max,
        )
        return

    selected_code = selected_patients[0]

    st.markdown("### Режим")
    mode_key = session_key("patient_mode", selected_code)
    mode = st.radio(
        "Выберите режим:",
        ["Все метаболиты", "Риски"],
        horizontal=True,
        label_visibility="collapsed",
        key=mode_key,
    )

    st.markdown("---")
    st.subheader(f"Пациент: {selected_code}")

    base_table = cached_patient_table(selected_code, sample, ref_min, ref_max)

    if mode == "Все метаболиты":
        render_table_and_plot(
            df=base_table,
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
    risk_key = session_key("risk", selected_code, "choice")
    previous_risk = st.session_state.get(risk_key, risk_names[0] if risk_names else "")
    risk_choice = st.selectbox(
        "Выберите группу риска:",
        risk_names,
        index=risk_names.index(previous_risk) if previous_risk in risk_names else 0,
        key=risk_key,
    )

    show_group_key = session_key("risk", selected_code, "show_full_group")
    show_full_group = st.checkbox(
        "Показать всю группу риска",
        value=st.session_state.get(show_group_key, False),
        key=show_group_key,
        help="Отобразить график со всеми метаболитами выбранной группы риска.",
    )

    risk_metabolites = risk_map.get(risk_choice, [])
    risk_df = base_table[base_table[METABOLITE_COLUMN].isin(risk_metabolites)].copy()

    st.markdown(f"#### {risk_choice}")
    if risk_df.empty:
        st.info("Для выбранного риска нет метаболитов в данных этого пациента.")
        return

    if show_full_group:
        render_risk_overview_chart(risk_df, selected_code, risk_choice)
        st.markdown("---")

    key_suffix = f"{risk_choice}_full" if show_full_group else risk_choice
    render_table_and_plot(
        df=risk_df,
        key_prefix=f"{selected_code}_risk_{key_suffix}",
        base_code=selected_code,
        sample=sample,
        ref_min=ref_min,
        ref_max=ref_max,
        patient_codes=patient_codes,
    )


if __name__ == "__main__":
    main()
    