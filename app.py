from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Breast Cancer Cohort | Explorer",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
EXPECTED_FILE = APP_DIR / "Breast_Cancer.csv"
REQUIRED_COLUMNS = [
    "Age", "Race", "Marital Status", "T Stage", "N Stage", "6th Stage",
    "differentiate", "Grade", "A Stage", "Tumor Size", "Estrogen Status",
    "Progesterone Status", "Regional Node Examined", "Reginol Node Positive",
    "Survival Months", "Status",
]
NUMERIC_COLUMNS = [
    "Age", "Tumor Size", "Regional Node Examined", "Reginol Node Positive",
    "Survival Months",
]
CAT_COLUMNS = [
    "Race", "Marital Status", "T Stage", "N Stage", "6th Stage", "Grade",
    "A Stage", "Estrogen Status", "Progesterone Status", "Status",
]
PALETTE = {"Alive": "#19a58c", "Dead": "#df6480"}
PLOT_TEMPLATE = "plotly_white"


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim column whitespace and normalize the notebook's common T Stage spelling."""
    frame = frame.copy()
    frame.columns = [str(col).strip() for col in frame.columns]
    if "T Stage" not in frame and "T Stage " in frame:
        frame = frame.rename(columns={"T Stage ": "T Stage"})
    # Accept the standard spelling as well as the typo used by the source notebook.
    if "Regional Node Positive" in frame and "Reginol Node Positive" not in frame:
        frame = frame.rename(columns={"Regional Node Positive": "Reginol Node Positive"})
    return frame


@st.cache_data(show_spinner=False)
def read_csv_bytes(contents: bytes) -> pd.DataFrame:
    import io

    return normalize_columns(pd.read_csv(io.BytesIO(contents)))


def prepare_data(frame: pd.DataFrame) -> pd.DataFrame:
    frame = normalize_columns(frame)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("Missing expected columns: " + ", ".join(missing))
    frame = frame.drop_duplicates().copy()
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["Grade"] = frame["Grade"].astype(str).str.strip()
    frame["Status"] = frame["Status"].astype(str).str.strip().str.title()
    frame["Age band"] = pd.cut(
        frame["Age"], bins=[0, 39, 49, 59, 69, np.inf],
        labels=["<40", "40–49", "50–59", "60–69", "70+"],
        right=True,
    )
    frame["Tumor size band"] = pd.cut(
        frame["Tumor Size"], bins=[0, 20, 50, np.inf],
        labels=["Small (≤20 mm)", "Medium (21–50 mm)", "Large (>50 mm)"],
        right=True,
    )
    frame["Positive node share"] = np.where(
        frame["Regional Node Examined"].gt(0),
        frame["Reginol Node Positive"] / frame["Regional Node Examined"],
        np.nan,
    )
    return frame


def style_chart(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=28, r=24, t=42, b=42),
        font=dict(family="Inter, Segoe UI, sans-serif", size=14, color="#243247"),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend_title_text="",
        hoverlabel=dict(bgcolor="white", bordercolor="#cbd5e1", font_size=14, font_color="#243247"),
        hovermode="closest",
    )
    fig.update_xaxes(
        showline=True, linecolor="#cbd5e1", linewidth=1,
        tickfont=dict(size=12, color="#526176"),
        title_font=dict(size=13, color="#34445a"),
        gridcolor="#edf1f5", zeroline=False,
    )
    fig.update_yaxes(
        showline=False,
        tickfont=dict(size=12, color="#526176"),
        title_font=dict(size=13, color="#34445a"),
        gridcolor="#edf1f5", zeroline=False,
    )
    return fig


def show_chart(fig: go.Figure, height: int = 390, *, selectable: bool = False):
    """Render a consistently sized chart with visible zoom, pan, and image controls."""
    return st.plotly_chart(
        style_chart(fig, height),
        use_container_width=True,
        config={
            "displaylogo": False,
            "displayModeBar": True,
            "scrollZoom": True,
            "responsive": True,
        },
        on_select="rerun" if selectable else "ignore",
        selection_mode=("points", "box", "lasso") if selectable else None,
    )


def pct(numerator: float, denominator: float) -> float:
    return float(numerator / denominator * 100) if denominator else 0.0


def fmt_months(value: Any) -> str:
    return "—" if pd.isna(value) else f"{value:.0f} mo"


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:#203044; --muted:#718096; --rose:#d85c79; --teal:#159b87; }
    html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(145deg,#f8fafc 0%,#f7f2f3 55%,#f3f8f7 100%); }
    .main .block-container { max-width: 1680px; padding-top: 1.15rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background: #fffaf9; border-right:1px solid #eee5e5; }
    .hero { padding: 1.45rem 1.7rem; border-radius:22px; color:white;
      background:linear-gradient(115deg,#26354b 0%,#55415d 55%,#a75670 100%);
      box-shadow:0 14px 38px rgba(42,45,67,.14); margin-bottom:1rem; }
    .hero h1 { font-family:Manrope,sans-serif; font-size:2rem; margin:0 0 .3rem 0; color:white; }
    .hero p { margin:0; color:#f1eaf0; font-size:.98rem; }
    .eyebrow { text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; font-weight:700; opacity:.82; }
    div[data-testid="stMetric"] { background:rgba(255,255,255,.88); padding:1rem 1.05rem;
      border:1px solid #ece8eb; border-radius:16px; box-shadow:0 5px 18px rgba(41,49,67,.045); }
    div[data-testid="stMetricLabel"] { color:#748195; }
    div[data-testid="stMetricValue"] { color:#26364a; font-family:Manrope,sans-serif; }
    .section-caption { color:#718096; margin-top:-.7rem; margin-bottom:.8rem; }
    .insight-card { background:#fff; border:1px solid #ece8eb; border-left:4px solid #d85c79;
      border-radius:12px; padding:.9rem 1rem; margin:.45rem 0; color:#435267; }
    .small-note { color:#718096; font-size:.84rem; }
    [data-testid="stTabs"] [role="tab"] { font-size:.96rem; font-weight:650; padding:.75rem 1rem; }
    [data-testid="stTabs"] [aria-selected="true"] { color:#a33d5b; }
    [data-testid="stPlotlyChart"] { background:#fff; border:1px solid #e8edf2; border-radius:14px;
      box-shadow:0 4px 16px rgba(35,48,68,.045); padding:.28rem; }
    [data-testid="stExpander"] { background:rgba(255,255,255,.72); border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Cohort explorer · interactive EDA</div>
      <h1>Breast cancer, understood in context</h1>
      <p>Explore the cohort’s recorded outcomes across age, tumor characteristics, stage, nodes, and receptor status.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🎗️ Cohort Explorer")
    st.caption("Choose a dataset, then refine every view with the shared filters.")
    uploaded = st.file_uploader("Upload Breast_Cancer.csv", type=["csv"], help="Expected: the 16 columns used in Breast_cancer_EDA.ipynb.")

    if uploaded is not None:
        try:
            raw = read_csv_bytes(uploaded.getvalue())
            source_label = uploaded.name
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
            st.stop()
    elif EXPECTED_FILE.exists():
        try:
            raw = normalize_columns(pd.read_csv(EXPECTED_FILE))
            source_label = EXPECTED_FILE.name
        except Exception as exc:
            st.error(f"Could not read {EXPECTED_FILE.name}: {exc}")
            st.stop()
    else:
        raw = None
        source_label = ""

if raw is None:
    st.info("Add the dataset to get started. Upload **Breast_Cancer.csv** in the sidebar, or place it beside `app.py` and refresh.")
    st.markdown(
        """
        <div class="insight-card"><b>What this dashboard explores</b><br>
        The notebook’s main questions: who is represented in the cohort, how recorded survival months and status vary by grade,
        tumor size and stage, and how many regional nodes were positive.</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("The notebook reports 4,024 records and 16 source fields. Its source file was referenced as `/content/Breast_Cancer.csv`, which is not included in this project folder.")
    st.stop()

try:
    data = prepare_data(raw)
except Exception as exc:
    st.error(f"This file does not match the notebook’s dataset schema. {exc}")
    st.stop()

with st.sidebar:
    st.success(f"Loaded {len(data):,} unique records · {source_label}")
    st.markdown("### Filters")
    quick_focus = st.selectbox(
        "Quick focus",
        ["All matching records", "Recorded alive", "Recorded dead", "Estrogen negative", "Stage IIIC"],
        help="Apply a common cohort focus in addition to the sliders and selections below.",
    )
    age_min = int(data["Age"].min())
    age_max = int(data["Age"].max())
    age_range = st.slider("Age (years)", age_min, age_max, (age_min, age_max))
    tumor_min = int(data["Tumor Size"].min())
    tumor_max = int(data["Tumor Size"].max())
    tumor_range = st.slider("Tumor size (mm)", tumor_min, tumor_max, (tumor_min, tumor_max))
    surv_min = int(data["Survival Months"].min())
    surv_max = int(data["Survival Months"].max())
    survival_range = st.slider("Recorded months", surv_min, surv_max, (surv_min, surv_max))
    selected_categories: dict[str, list[str]] = {}
    with st.expander("Demographics", expanded=False):
        for col in ["Race", "Marital Status"]:
            options = sorted(data[col].dropna().astype(str).unique().tolist())
            selected_categories[col] = st.multiselect(col, options, default=options, key=f"filter_{col}")
    with st.expander("Cancer characteristics", expanded=False):
        clinical_columns = [column for column in CAT_COLUMNS if column not in {"Race", "Marital Status"}]
        for col in clinical_columns:
            options = sorted(data[col].dropna().astype(str).unique().tolist())
            selected_categories[col] = st.multiselect(col, options, default=options, key=f"filter_{col}")

filtered = data[
    data["Age"].between(age_range[0], age_range[1])
    & data["Tumor Size"].between(tumor_range[0], tumor_range[1])
    & data["Survival Months"].between(survival_range[0], survival_range[1])
].copy()
for col, choices in selected_categories.items():
    if choices:
        filtered = filtered[filtered[col].astype(str).isin(choices)]
    else:
        filtered = filtered.iloc[0:0]
if quick_focus == "Recorded alive":
    filtered = filtered[filtered["Status"] == "Alive"]
elif quick_focus == "Recorded dead":
    filtered = filtered[filtered["Status"] == "Dead"]
elif quick_focus == "Estrogen negative":
    filtered = filtered[filtered["Estrogen Status"] == "Negative"]
elif quick_focus == "Stage IIIC":
    filtered = filtered[filtered["6th Stage"] == "IIIC"]

if filtered.empty:
    st.warning("No records match these filters. Broaden one or more selections in the sidebar.")
    st.stop()

st.markdown(f"<div class='small-note'>Showing <b>{len(filtered):,}</b> of {len(data):,} unique records · source: {source_label}</div>", unsafe_allow_html=True)
overview_tab, patterns_tab, relationships_tab, records_tab, quality_tab = st.tabs(
    ["✦ Overview", "◉ Clinical patterns", "⌁ Relationships", "▤ Patient explorer", "✓ Data quality"]
)

with overview_tab:
    n = len(filtered)
    dead = int(filtered["Status"].eq("Dead").sum())
    alive = int(filtered["Status"].eq("Alive").sum())
    kpis = st.columns(5)
    kpis[0].metric("Records in view", f"{n:,}")
    kpis[1].metric("Recorded alive", f"{alive:,}", f"{pct(alive, n):.1f}% of view")
    kpis[2].metric("Recorded dead", f"{dead:,}", f"{pct(dead, n):.1f}% crude share")
    kpis[3].metric("Median age", f"{filtered['Age'].median():.0f} yr")
    kpis[4].metric("Median recorded months", fmt_months(filtered["Survival Months"].median()))
    st.caption("Status is the dataset’s recorded label. The crude dead share is descriptive and is not a survival probability or a causal estimate.")

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.subheader("Who is represented?")
        ages = filtered.groupby("Age band", observed=False).size().rename("Patients").reset_index()
        fig = px.bar(ages, x="Age band", y="Patients", color="Age band", text_auto=".2s", color_discrete_sequence=px.colors.sequential.Tealgrn)
        fig.update_layout(showlegend=False, xaxis_title="Age band (years)", yaxis_title="Patients")
        fig.update_traces(textposition="outside")
        show_chart(fig, 390)
    with right:
        st.subheader("Recorded status across stages")
        stage_order = [x for x in ["IIA", "IIB", "IIIA", "IIIB", "IIIC"] if x in filtered["6th Stage"].astype(str).unique()]
        stage_df = filtered.copy()
        stage_df["6th Stage"] = pd.Categorical(stage_df["6th Stage"].astype(str), categories=stage_order, ordered=True)
        stage_counts = stage_df.groupby(["6th Stage", "Status"], observed=False).size().rename("Patients").reset_index()
        fig = px.bar(stage_counts, x="6th Stage", y="Patients", color="Status", barmode="group", color_discrete_map=PALETTE)
        fig.update_layout(xaxis_title="6th stage", yaxis_title="Patients")
        show_chart(fig, 390)

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Recorded months by status")
        fig = px.violin(filtered, x="Status", y="Survival Months", color="Status", box=True, points="outliers", color_discrete_map=PALETTE)
        fig.update_layout(xaxis_title="Recorded status", yaxis_title="Months")
        show_chart(fig, 390)
    with right:
        st.subheader("Tumor size distribution")
        tumor_dist = filtered.groupby("Tumor size band", observed=False).size().rename("Patients").reset_index()
        fig = px.pie(tumor_dist, names="Tumor size band", values="Patients", hole=.62, color_discrete_sequence=["#52ad9a", "#8297b7", "#d96d83"])
        fig.update_traces(textposition="outside", textinfo="label+percent")
        show_chart(fig, 390)

    st.subheader("At a glance")
    largest_age = filtered["Age band"].value_counts().idxmax()
    largest_stage = filtered["6th Stage"].astype(str).value_counts().idxmax()
    largest_grade = filtered["Grade"].astype(str).value_counts().idxmax()
    age_counts = filtered.groupby("Age band", observed=True).size()
    tumor_insight = filtered.assign(_dead=filtered["Status"].eq("Dead").astype(int)).groupby(
        "Tumor size band", observed=True
    ).agg(records=("Status", "size"), dead_share=("_dead", "mean"))
    stage_insight = filtered.groupby("6th Stage", observed=True)["Reginol Node Positive"].mean()
    grade_insight = filtered.assign(_dead=filtered["Status"].eq("Dead").astype(int)).groupby(
        "Grade", observed=True
    ).agg(records=("Status", "size"), dead_share=("_dead", "mean"))
    insight_cols = st.columns(4, gap="medium")
    with insight_cols[0]:
        st.metric("Largest age band", str(largest_age), f"{pct(age_counts[largest_age], n):.1f}% of view")
        st.caption("Age distribution")
    with insight_cols[1]:
        if len(tumor_insight) >= 2:
            small = tumor_insight.iloc[0]
            large = tumor_insight.iloc[-1]
            st.metric("Tumor size pattern", f"{small['dead_share'] * 100:.1f}% → {large['dead_share'] * 100:.1f}%", "crude recorded-dead share · small to large")
        else:
            st.metric("Tumor size pattern", "—", "Select multiple size bands to compare")
        st.caption("Descriptive comparison")
    with insight_cols[2]:
        if len(stage_insight) >= 2:
            st.metric("Positive nodes by stage", f"{stage_insight.iloc[0]:.1f} → {stage_insight.iloc[-1]:.1f}", f"mean · {stage_insight.index[0]} to {stage_insight.index[-1]}")
        else:
            st.metric("Positive nodes by stage", "—", "Select multiple stages to compare")
        st.caption("Regional node involvement")
    with insight_cols[3]:
        if "1" in grade_insight.index.astype(str) and "3" in grade_insight.index.astype(str):
            grade1 = grade_insight.loc[grade_insight.index.astype(str) == "1"].iloc[0]
            grade3 = grade_insight.loc[grade_insight.index.astype(str) == "3"].iloc[0]
            st.metric("Grade 1 → grade 3", f"{grade1['dead_share'] * 100:.1f}% → {grade3['dead_share'] * 100:.1f}%", "crude recorded-dead share")
        else:
            st.metric("Most common grade", str(largest_grade), f"{pct((filtered['Grade'].astype(str) == str(largest_grade)).sum(), n):.1f}% of view")
        st.caption("Descriptive comparison")
    st.caption("Insights recalculate when filters change. Recorded-dead shares are cohort descriptions, not adjusted risk estimates.")

with patterns_tab:
    st.subheader("Clinical patterns in the filtered cohort")
    st.markdown("<div class='section-caption'>Compare observed group summaries. Group sizes are shown in hover details where available.</div>", unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("#### Months by grade and status")
        grade_order = sorted(filtered["Grade"].dropna().astype(str).unique(), key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
        fig = px.box(filtered, x="Grade", y="Survival Months", color="Status", category_orders={"Grade": grade_order}, points=False, color_discrete_map=PALETTE)
        fig.update_layout(xaxis_title="Grade", yaxis_title="Recorded months")
        show_chart(fig, 420)
    with b:
        st.markdown("#### Tumor size by status")
        fig = px.box(filtered, x="Status", y="Tumor Size", color="Status", points="outliers", color_discrete_map=PALETTE)
        fig.update_layout(xaxis_title="Recorded status", yaxis_title="Tumor size (mm)")
        show_chart(fig, 420)

    c, d = st.columns(2, gap="large")
    with c:
        st.markdown("#### Average positive nodes by stage")
        node_summary = filtered.groupby(["6th Stage", "Status"], observed=True).agg(
            mean_positive_nodes=("Reginol Node Positive", "mean"), patients=("Status", "size")
        ).reset_index()
        fig = px.bar(node_summary, x="6th Stage", y="mean_positive_nodes", color="Status", barmode="group", color_discrete_map=PALETTE, hover_data={"patients": True, "mean_positive_nodes": ":.2f"})
        fig.update_layout(xaxis_title="6th stage", yaxis_title="Mean positive regional nodes")
        show_chart(fig, 400)
    with d:
        st.markdown("#### Crude recorded-dead share by group")
        breakdown = st.selectbox("Break down by", ["Age band", "Tumor size band", "Grade", "6th Stage", "Estrogen Status", "Progesterone Status"], key="mortality_breakdown")
        rates = filtered.assign(_dead=filtered["Status"].eq("Dead").astype(int)).groupby(breakdown, observed=True).agg(
            records=("Status", "size"), recorded_dead_share=("_dead", "mean")
        ).reset_index()
        rates["recorded_dead_share"] *= 100
        fig = px.bar(rates, x=breakdown, y="recorded_dead_share", color="recorded_dead_share", color_continuous_scale="Tealrose", text=rates["recorded_dead_share"].map(lambda x: f"{x:.1f}%"), hover_data={"records": True, "recorded_dead_share": ":.1f"})
        fig.update_layout(xaxis_title=breakdown, yaxis_title="Recorded dead share (%)", coloraxis_showscale=False)
        fig.update_traces(textposition="outside")
        show_chart(fig, 400)

    st.markdown("#### Build a comparison")
    controls = st.columns([1, 1, 1.4], gap="medium")
    metric_name = controls[0].selectbox(
        "Measure",
        ["Crude recorded-dead share", "Median recorded months", "Mean positive nodes", "Median tumor size"],
        key="custom_measure",
    )
    group_column = controls[1].selectbox(
        "Group patients by",
        ["6th Stage", "Age band", "Tumor size band", "Grade", "Race", "Estrogen Status", "Progesterone Status", "Marital Status"],
        key="custom_group",
    )
    if metric_name == "Crude recorded-dead share":
        compare = filtered.assign(_dead=filtered["Status"].eq("Dead").astype(int)).groupby(group_column, observed=True).agg(
            records=("Status", "size"), value=("_dead", "mean")
        ).reset_index()
        compare["value"] *= 100
        axis_label = "Recorded-dead share (%)"
        text_format = ".1f"
    else:
        measure_column, agg = {
            "Median recorded months": ("Survival Months", "median"),
            "Mean positive nodes": ("Reginol Node Positive", "mean"),
            "Median tumor size": ("Tumor Size", "median"),
        }[metric_name]
        compare = filtered.groupby(group_column, observed=True).agg(
            records=("Status", "size"), value=(measure_column, agg)
        ).reset_index()
        axis_label = {
            "Median recorded months": "Median recorded months",
            "Mean positive nodes": "Mean positive regional nodes",
            "Median tumor size": "Median tumor size (mm)",
        }[metric_name]
        text_format = ".1f"
    fig = px.bar(
        compare,
        x="value", y=group_column, orientation="h", color="value",
        color_continuous_scale=[[0, "#b9e3db"], [1, "#087f78"]],
        text=compare["value"].map(lambda value: format(value, text_format)),
        hover_data={"records": True, "value": ":.2f"},
    )
    fig.update_layout(
        xaxis_title=axis_label,
        yaxis_title="",
        coloraxis_showscale=False,
        yaxis={"categoryorder": "total ascending"},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    controls[2].caption("Updates instantly with the shared filters. Hover for the number of records in each group.")
    show_chart(fig, max(390, min(560, 180 + 45 * len(compare))))
    st.caption("The recorded-dead share compares labels in this dataset; it does not adjust for follow-up time, censoring, treatment, or other clinical factors.")

with relationships_tab:
    st.subheader("How numeric measures move together")
    c1, c2 = st.columns([1, 1.2], gap="large")
    corr_cols = NUMERIC_COLUMNS
    corr = filtered[corr_cols].corr(numeric_only=True)
    with c1:
        st.markdown("#### Correlation map")
        fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto")
        fig.update_layout(coloraxis_colorbar_title="Pearson r")
        show_chart(fig, 470)
    with c2:
        st.markdown("#### Explore a relationship")
        choices = ["Age", "Tumor Size", "Regional Node Examined", "Reginol Node Positive", "Survival Months"]
        axis_controls = st.columns(2)
        x_col = axis_controls[0].selectbox("Horizontal axis", choices, index=0, key="rel_x")
        y_choices = [column for column in choices if column != x_col]
        y_col = axis_controls[1].selectbox("Vertical axis", y_choices, index=y_choices.index("Survival Months") if "Survival Months" in y_choices else 0, key="rel_y")
        style_controls = st.columns(2)
        hue_col = style_controls[0].selectbox("Color points by", ["Status", "6th Stage", "Grade", "Estrogen Status"], key="rel_color")
        size_choice = style_controls[1].selectbox("Point size", ["Same size", "Tumor Size", "Reginol Node Positive", "Age"], key="rel_size")
        marker_size = None if size_choice == "Same size" else size_choice
        scatter_colors = PALETTE if hue_col == "Status" else px.colors.qualitative.Safe
        detail_columns = ["Age", "6th Stage", "Grade", "Status", "Tumor Size", "Survival Months"]
        fig = px.scatter(
            filtered, x=x_col, y=y_col, color=hue_col, size=marker_size,
            size_max=18, custom_data=detail_columns,
            opacity=.72, color_discrete_map=scatter_colors if hue_col == "Status" else None,
            color_discrete_sequence=None if hue_col == "Status" else scatter_colors,
        )
        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, legend_title=hue_col)
        chart_selection = show_chart(fig, 420, selectable=True)
        selected_points = chart_selection.selection.points
        if selected_points:
            selected_rows = [point.get("customdata") for point in selected_points if point.get("customdata")]
            if selected_rows:
                st.caption(f"{len(selected_rows)} selected points · box or lasso select in the chart; double-click to reset zoom.")
                st.dataframe(pd.DataFrame(selected_rows, columns=detail_columns), use_container_width=True, hide_index=True)
        else:
            st.caption("Drag to zoom · use the toolbar for pan, box zoom, lasso selection, and image export.")
    corr_pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    if not corr_pairs.empty:
        strongest = corr_pairs.abs().idxmax()
        st.markdown(f"<div class='insight-card'>Strongest numeric pair in this view: <b>{strongest[0]}</b> and <b>{strongest[1]}</b> (Pearson r = {corr_pairs.loc[strongest]:.2f}). Correlation describes linear association only.</div>", unsafe_allow_html=True)

with records_tab:
    st.subheader("Find and export cohort records")
    search = st.text_input("Search visible records", placeholder="Try a stage, grade, race, or status…")
    default_columns = ["Age", "Race", "6th Stage", "Grade", "Tumor Size", "Estrogen Status", "Reginol Node Positive", "Survival Months", "Status"]
    visible_columns = st.multiselect(
        "Columns to display",
        REQUIRED_COLUMNS,
        default=default_columns,
        help="Choose a smaller set of columns to keep the table easy to scan. The download still contains all source columns.",
    )
    table = filtered.copy()
    if search.strip():
        mask = table.astype(str).apply(lambda col: col.str.contains(search.strip(), case=False, regex=False)).any(axis=1)
        table = table[mask]
    st.caption(f"{len(table):,} matching records")
    display_columns = REQUIRED_COLUMNS
    if not visible_columns:
        st.info("Select at least one column to display the table.")
    else:
        st.dataframe(table[visible_columns], use_container_width=True, hide_index=True, height=520)
    csv_bytes = table[display_columns].to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download these records as CSV", csv_bytes, file_name="breast_cancer_filtered_records.csv", mime="text/csv")

with quality_tab:
    st.subheader("Dataset quality and definitions")
    st.markdown("<div class='section-caption'>Checks are calculated on the loaded source file before dashboard filters.</div>", unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Unique records", f"{len(data):,}")
    q2.metric("Duplicate rows removed", f"{len(raw) - len(data):,}")
    q3.metric("Missing cells", f"{int(data[REQUIRED_COLUMNS].isna().sum().sum()):,}")
    q4.metric("Source fields", f"{len(REQUIRED_COLUMNS)}")
    quality = pd.DataFrame({
        "Field": REQUIRED_COLUMNS,
        "Type": [str(data[col].dtype) for col in REQUIRED_COLUMNS],
        "Missing": [int(data[col].isna().sum()) for col in REQUIRED_COLUMNS],
        "Missing %": [round(float(data[col].isna().mean() * 100), 2) for col in REQUIRED_COLUMNS],
        "Unique values": [int(data[col].nunique(dropna=True)) for col in REQUIRED_COLUMNS],
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)
    st.markdown("#### Reading this dataset")
    st.markdown(
        "- The notebook defines age and tumor-size bands for descriptive plots; this app derives matching bands and makes their cutoffs explicit.\n"
        "- ‘Survival Months’ and ‘Status’ are shown as recorded fields. No time-to-event model is fitted.\n"
        "- ‘Reginol Node Positive’ preserves the source column spelling. ‘Positive node share’ is the positive nodes divided by nodes examined, when the denominator is above zero.\n"
        "- This is an exploratory dataset view, not a clinical decision or patient-level prognosis tool."
    )

st.markdown("---")
st.caption("Breast cancer cohort EDA · Values are descriptive summaries of the loaded dataset.")
