# Streamlit Metabolite Dashboard

An interactive **Streamlit-based analytical platform** designed to visualize the dynamics of metabolites and deviations from the norm in patient samples over multiple time points.

The application is intended for biomedical and bioinformatics research projects involving metabolomic profiling and patient monitoring.

---

## Background

Metabolomic data often includes measurements taken at multiple time points **time points**, **patient groups**, and **reference ranges** that need to be compared interactively.  
The dashboard offers a user-friendly, reproducible, and easy-to-understand visualization interface for biomedical researchers.

---

## Features

- **Upload Excel files** with patient data and reference values (`Ref.xlsx`).
- **Automatic comparison** of metabolite levels with reference intervals.
- **Color-coded visualization**:
  - 🔵 (blue) below normal  
  - 🔴 (red) above normal  
  - ⚪ (gray) within range
- **Interactive tables** using `st-aggrid` with filtering, sorting, and highlighting.
- **Dynamic Plotly charts** for:
  - per-patient longitudinal visualization,  
  - multi-patient comparison,  
  - time-course dynamics (Time-point 1–3).
- **Selection modes**: “All metabolites”, “By risk groups”, “Individual comparison”.

---

## Run the Application
After installation, launch the Streamlit dashboard:
streamlit run app.py
Then open your browser at:
http://localhost:8501

---

## Project Structure
```

streamlit_app/
│
├── app.py               # Streamlit controller orchestrating views
├── Ref.xlsx             # Reference ranges for metabolites
├── requirements.txt     # Python dependencies
├── README.md            # Project description and usage
└── src/
    ├── __init__.py      # Marks src/ as a Python package to enable clean imports
    ├── config.py        # Central constants, color palette, enums, file paths
    ├── data.py          # File loading (Ref.xlsx and patient data), normalization helpers, session keys
    ├── services.py      # Core business logic: PatientSnapshot, caching, sorting, chart payload prep
    └── ui.py            # AgGrid + Plotly rendering utilities used by app.py
```
