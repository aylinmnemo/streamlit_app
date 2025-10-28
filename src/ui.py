from typing import Any, Iterable, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative as plotly_qual
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

from src.config import (
    AGGRID_CUSTOM_CSS,
    AGGRID_DEFAULT_HEIGHT,
    AGGRID_ON_CELL_CLICK_JS,
    AGGRID_THEME,
    BASE_LINE_COLOR,
    COLOR_BG_ABOVE,
    COLOR_BG_BELOW,
    COLOR_BG_MISSING,
    COLOR_BG_NORMAL,
    COMPARISON_BUTTON_LABEL,
    METABOLITE_COLUMN,
    NORMS_COLUMN,
    PLACEHOLDER,
    REF_MAX_COLUMN,
    REF_MIN_COLUMN,
    SORT_MODES,
    TIMEPOINTS,
    TIMEPOINT_CELL_STYLE_JSON,
    TIMEPOINT_COLUMN_WIDTH,
    METABOLITE_CELL_STYLE_JSON,
    NORMS_CELL_STYLE_JSON,
    SELECTION_OVERLAY,
    AGGRID_COLUMN_WIDTHS,
    DEFAULT_FIG_HEIGHT,
    REQUIRED_SAMPLE_COLUMNS,
    BASE_CODE_COLUMN,
    TP_JS_MAP_JSON,
    RAW_COLUMNS,
)
from src.data import display_value_precise, session_key, to_float
from src.services import (
    cached_patient_table,
    chart_axis_limits,
    collect_comparison_lines,
    collect_numeric_values,
    marker_colors,
    metabolite_chart_payload,
    patient_metabolite_line,
    prepare_display_dataframe,
    reference_bands,
    sort_by_status,
)


# AgGrid helpers

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
          if (raw < rmin - 1e-12) baseColor = '#b3d1ff';
          else if (raw > rmax + 1e-12) baseColor = '#ff9999';
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


def build_grid_options(df_disp: pd.DataFrame, on_click: JsCode, color_cells: JsCode) -> dict:
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
    css_key = "__aggrid_css_applied"
    if st.session_state.get(css_key):
        return
    st.session_state[css_key] = True
    st.markdown(AGGRID_CUSTOM_CSS, unsafe_allow_html=True)


def render_table_controls(key_prefix: str) -> tuple[str, bool]:
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


def render_grid_and_get_selection(
    df_disp: pd.DataFrame,
    grid_options: dict,
    key_prefix: str,
    show_table: bool,
    selected_key: str,
    current_selection: str | None,
) -> tuple[str | None, Any]:
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
            comp_df = cached_patient_table(entry["patient"], sample, ref_min, ref_max)
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


def render_comparison_chart(
    df: pd.DataFrame,
    metabolite: str,
    base_code: str,
    comparisons: list[dict[str, str]],
    sample: pd.DataFrame,
    ref_min: pd.Series,
    ref_max: pd.Series,
) -> None:
    payload = metabolite_chart_payload(df, metabolite, ref_min, ref_max)
    if payload is None:
        st.info("Выберите метаболит в таблице.")
        return
    base_values, rmin, rmax, marker_cols = payload

    comparison_lines = collect_comparison_lines(comparisons, sample, ref_min, ref_max)
    numeric_values = collect_numeric_values(base_values, comparison_lines)
    y0, y1 = chart_axis_limits(rmin, rmax, numeric_values)

    fig = go.Figure()
    for band in reference_bands(y0, rmin, rmax, y1):
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
                color=marker_cols,
                size=10,
                symbol="circle",
                line=dict(width=1, color="rgba(0,0,0,0.35)"),
            ),
        )
    )

    palette = plotly_qual.Plotly
    for idx, line in enumerate(comparison_lines):
        color = palette[idx % len(palette)]
        comp_marker_colors = marker_colors(line["values"], rmin, rmax)
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


def render_risk_overview_chart(
    df: pd.DataFrame,
    base_code: str,
    risk_label: str,
) -> None:
    if df.empty:
        st.info("Нет данных для построения графика риска.")
        return

    palette = plotly_qual.Plotly
    fig = go.Figure()
    numeric_values: list[float] = []
    ref_min_values: list[float] = []
    ref_max_values: list[float] = []
    traces_added = 0

    for idx, row in df.iterrows():
        metabolite = row[METABOLITE_COLUMN]
        values = [to_float(row[tp]) for tp in TIMEPOINTS]
        if all(pd.isna(value) for value in values):
            continue

        rmin = to_float(row[REF_MIN_COLUMN])
        rmax = to_float(row[REF_MAX_COLUMN])
        marker_cols = marker_colors(values, rmin, rmax)

        numeric_values.extend(value for value in values if pd.notna(value))
        if pd.notna(rmin):
            ref_min_values.append(rmin)
        if pd.notna(rmax):
            ref_max_values.append(rmax)

        color = palette[traces_added % len(palette)]
        traces_added += 1

        fig.add_trace(
            go.Scatter(
                x=list(TIMEPOINTS),
                y=values,
                mode="lines+markers",
                name=metabolite,
                hoverinfo="x+y+name",
                line=dict(color=color, dash="solid"),
                marker=dict(
                    color=marker_cols,
                    size=9,
                    symbol="circle",
                    line=dict(width=1, color="rgba(0,0,0,0.35)"),
                ),
            )
        )

    if traces_added == 0 or not numeric_values:
        st.info("Нет измерений для выбранного риска.")
        return

    global_min = min(ref_min_values) if ref_min_values else min(numeric_values)
    global_max = max(ref_max_values) if ref_max_values else max(numeric_values)
    y0, y1 = chart_axis_limits(global_min, global_max, numeric_values)

    for band in reference_bands(y0, global_min, global_max, y1):
        fig.add_hrect(**band)

    fig.update_layout(
        title=dict(text=f"{risk_label} · {base_code}", y=0.96, yanchor="top"),
        template="simple_white",
        height=DEFAULT_FIG_HEIGHT,
        margin=dict(l=10, r=190, t=70, b=60),
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


def render_multi_patient_comparison(
    metabolite: str,
    patients: Sequence[str],
    sample: pd.DataFrame,
    ref_min: pd.Series,
    ref_max: pd.Series,
) -> None:
    if not patients:
        st.info("Выберите пациентов для сравнения.")
        return

    lines: list[dict[str, Any]] = []
    for code in patients:
        line = patient_metabolite_line(code, metabolite, sample, ref_min, ref_max)
        if line:
            lines.append(line)

    if not lines:
        st.info("Для выбранного метаболита нет значений у отмеченных пациентов.")
        return

    numeric_values = [value for line in lines for value in line["values"] if pd.notna(value)]
    valid_bounds = [
        (line["rmin"], line["rmax"])
        for line in lines
        if pd.notna(line["rmin"]) and pd.notna(line["rmax"])
    ]
    if valid_bounds:
        rmin = min(bound[0] for bound in valid_bounds)
        rmax = max(bound[1] for bound in valid_bounds)
    elif numeric_values:
        rmin = min(numeric_values)
        rmax = max(numeric_values)
    else:
        rmin, rmax = 0.0, 1.0

    y0, y1 = chart_axis_limits(rmin, rmax, numeric_values)
    summary_rows: list[dict[str, str]] = []
    background_rows: list[dict[str, str]] = []

    for line in lines:
        row: dict[str, str] = {"Пациент": line["patient"]}
        backgrounds: dict[str, str] = {}
        line_rmin, line_rmax = line["rmin"], line["rmax"]
        for idx, tp in enumerate(TIMEPOINTS):
            value = line["values"][idx]
            display = display_value_precise(value, line_rmin, line_rmax) or PLACEHOLDER
            row[tp] = display
            if pd.isna(value):
                background = COLOR_BG_MISSING
            elif pd.notna(line_rmin) and value < line_rmin - 1e-12:
                background = COLOR_BG_BELOW
            elif pd.notna(line_rmax) and value > line_rmax + 1e-12:
                background = COLOR_BG_ABOVE
            else:
                background = COLOR_BG_NORMAL
            backgrounds[tp] = background

        if pd.notna(line_rmin) and pd.notna(line_rmax):
            row[NORMS_COLUMN] = f"{line_rmin:g} - {line_rmax:g}"
        else:
            row[NORMS_COLUMN] = PLACEHOLDER

        summary_rows.append(row)
        background_rows.append(backgrounds)

    summary_df = pd.DataFrame(summary_rows)

    def _apply_background(row: pd.Series) -> list[str]:
        idx = row.name
        bg_map = background_rows[idx]
        styles: list[str] = []
        for column in row.index:
            if column in TIMEPOINTS:
                styles.append(f"background-color: {bg_map.get(column, COLOR_BG_NORMAL)};")
            else:
                styles.append("")
        return styles

    table_col, chart_col = st.columns([7, 9], gap="medium")

    with table_col:
        styler = (
            summary_df.style.apply(_apply_background, axis=1)
            .set_properties(subset=["Пациент"], **{"text-align": "left"})
            .set_properties(subset=list(TIMEPOINTS), **{"text-align": "center"})
            .set_properties(subset=[NORMS_COLUMN], **{"text-align": "center", "font-style": "italic"})
        )
        st.dataframe(styler, use_container_width=True)

    with chart_col:
        fig = go.Figure()

        if valid_bounds:
            for band in reference_bands(y0, rmin, rmax, y1):
                fig.add_hrect(**band)

        palette = plotly_qual.Plotly
        for idx, line in enumerate(lines):
            color = palette[idx % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=list(TIMEPOINTS),
                    y=line["values"],
                    mode="lines+markers",
                    name=line["patient"],
                    line=dict(color=color, dash="solid"),
                    hoverinfo="x+y+name",
                    marker=dict(
                        color=line["colors"],
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
    if df.empty:
        st.warning("Нет данных для отображения.")
        return

    selected_key = session_key("selected_metabolite", key_prefix)
    clicked = st.session_state.get(selected_key)

    sort_mode, show_table = render_table_controls(key_prefix)
    sorted_df = sort_by_status(df, sort_mode)
    df_disp = prepare_display_dataframe(sorted_df)

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

    if clicked not in set(sorted_df[METABOLITE_COLUMN]):
        if sorted_df.empty:
            clicked = None
        else:
            clicked = sorted_df.iloc[0][METABOLITE_COLUMN]
            st.session_state[selected_key] = clicked

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


def render_multi_patient_mode(
    selected_patients: Sequence[str],
    sample: pd.DataFrame,
    ref_min: pd.Series,
    ref_max: pd.Series,
) -> None:
    if not selected_patients:
        st.info("Отметьте пациентов слева, чтобы построить сравнение.")
        return

    candidate_metabolites = [
        column
        for column in sample.columns
        if column not in REQUIRED_SAMPLE_COLUMNS
        and column != BASE_CODE_COLUMN
        and column in ref_min.index
        and column in ref_max.index
    ]
    metabolite_options = sorted(
        {str(name).strip() for name in candidate_metabolites if str(name).strip()}
    )
    if not metabolite_options:
        st.warning("Не найдено метаболитов, доступных одновременно в данных и в референсах.")
        return

    metabolite_key = session_key("multi_compare", "metabolite")
    default_metabolite = st.session_state.get(metabolite_key, metabolite_options[0])
    if default_metabolite not in metabolite_options:
        default_metabolite = metabolite_options[0]
        st.session_state[metabolite_key] = default_metabolite

    selected_metabolite = st.selectbox(
        "Метаболит:",
        metabolite_options,
        index=metabolite_options.index(default_metabolite),
        key=metabolite_key,
    )

    st.markdown("---")
    render_multi_patient_comparison(
        metabolite=selected_metabolite,
        patients=selected_patients,
        sample=sample,
        ref_min=ref_min,
        ref_max=ref_max,
    )
