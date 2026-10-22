"""Dashboard profesional — Taller 2: Screener financiero de aerolíneas.

Alineado con el notebook final del Taller 2:
- Ratios de las cuatro familias
- DuPont de 3 y 5 factores + atribución del cambio del ROE
- Motores operacionales
- Common-size y tendencias
- Peer screening, ranking y recomendación
- Accruals, Beneish M-Score y Altman Z / Z''

Ejecución local:
    pip install streamlit plotly pandas numpy
    streamlit run app_dashboard_mejorado_compatible.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# CONFIGURACIÓN VISUAL
# =============================================================================
NAVY = "#17365D"
BLUE = "#2E5496"
ACCENT = "#C55A11"
GREY = "#7B8491"
LIGHT = "#EEF2F7"
GREEN = "#2E7D57"
RED = "#B23A3A"
AMBER = "#C9A227"
DARK = "#1F2937"
WHITE = "#FFFFFF"

st.set_page_config(
    page_title="Taller 2 | Screener financiero de aerolíneas",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
        .block-container {{padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px;}}
        [data-testid="stSidebar"] {{background-color: #F7F9FC; border-right: 1px solid #E4E9F0;}}
        h1, h2, h3 {{color: {NAVY}; letter-spacing: -0.02em;}}
        h1 {{font-size: 2.05rem !important; margin-bottom: .15rem !important;}}
        h2 {{font-size: 1.45rem !important; margin-top: 1.2rem !important;}}
        h3 {{font-size: 1.08rem !important;}}
        div[data-testid="stMetric"] {{
            background: white;
            border: 1px solid #E3E8EF;
            border-radius: 12px;
            padding: 0.8rem 0.9rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        }}
        div[data-testid="stMetricLabel"] {{color: {GREY}; font-weight: 600;}}
        div[data-testid="stMetricValue"] {{color: {NAVY};}}
        .hero {{
            padding: 1.15rem 1.3rem;
            border-radius: 14px;
            background: linear-gradient(135deg, #F7F9FC 0%, #EEF3F9 100%);
            border: 1px solid #DDE5EF;
            margin-bottom: 1rem;
        }}
        .hero-kicker {{font-size: .78rem; font-weight: 700; color: {ACCENT}; text-transform: uppercase; letter-spacing: .08em;}}
        .hero-title {{font-size: 1.45rem; font-weight: 750; color: {NAVY}; margin: .15rem 0 .25rem 0;}}
        .hero-copy {{font-size: .96rem; color: #455468; margin: 0;}}
        .pill {{
            display:inline-block; padding:.18rem .55rem; margin:.12rem .15rem .12rem 0;
            border-radius:999px; font-size:.78rem; font-weight:650; background:#E8EEF6; color:{NAVY};
        }}
        .section-note {{
            padding:.75rem .9rem; border-left: 4px solid {ACCENT}; background:#FCF7F2;
            border-radius: 0 8px 8px 0; margin:.55rem 0 1rem 0; color:#4A5568;
        }}
        .small-muted {{font-size:.82rem; color:{GREY};}}
        .rank-card {{background:white; border:1px solid #E3E8EF; border-radius:12px; padding:.75rem .9rem;}}
        [data-testid="stDataFrame"] {{border: 1px solid #E5EAF0; border-radius: 10px; overflow: hidden;}}
        .stTabs [data-baseweb="tab-list"] {{gap: .25rem;}}
        .stTabs [data-baseweb="tab"] {{font-weight: 650;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# DATOS Y CÁLCULOS BASE
# =============================================================================
@st.cache_data
def cargar(archivo) -> pd.DataFrame:
    """Carga CSV subido o panel_para_dashboard.csv ubicado junto al script."""
    if archivo is not None:
        df = pd.read_csv(archivo)
    else:
        carpeta = Path(__file__).resolve().parent
        candidatos = [
            carpeta / "panel_para_dashboard.csv",
            carpeta / "fundamentals_panel_respaldo.csv",
        ]
        fuente = next((p for p in candidatos if p.exists()), None)
        if fuente is None:
            return pd.DataFrame()
        df = pd.read_csv(fuente)

    if "anio" in df.columns:
        df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")
    return df.sort_values(["empresa", "anio"]).reset_index(drop=True)


def div(a, b):
    out = a / b
    if isinstance(out, pd.Series):
        return out.replace([np.inf, -np.inf], np.nan)
    return np.nan if not np.isfinite(out) else out


def ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Mismos ratios utilizados en el Taller 2."""
    r = df[["empresa", "anio"]].copy()

    # Liquidez
    r["Razón corriente"] = div(df.current_assets, df.current_liabilities)
    r["Prueba ácida"] = div(df.current_assets - df.inventory, df.current_liabilities)

    # Rentabilidad
    r["Margen bruto"] = div(df.gross_profit, df.revenue)
    r["Margen operacional"] = div(df.ebit, df.revenue)
    r["Margen neto"] = div(df.net_income, df.revenue)
    r["ROA"] = div(df.net_income, df.total_assets)
    r["ROE"] = div(df.net_income, df.equity)

    # Endeudamiento
    r["Deuda/Patrimonio"] = div(df.total_liabilities, df.equity)
    r["Cobertura de intereses"] = div(df.ebit, df.interest_expense)

    # Eficiencia
    r["Rotación de activos"] = div(df.revenue, df.total_assets)
    r["DSO (días)"] = 365 * div(df.receivables, df.revenue)
    r["Días de inventario"] = 365 * div(df.inventory, df.cogs)
    r["Ventas/PP&E"] = div(df.revenue, df.ppe)

    # DuPont
    r["Multiplicador de capital"] = div(df.total_assets, df.equity)
    uai = df.ebit - df.interest_expense
    if "pretax_income" in df.columns:
        uai = df["pretax_income"].fillna(uai)
    r["Carga tributaria"] = div(df.net_income, uai)
    r["Carga de intereses"] = div(uai, df.ebit)
    r["Margen EBIT"] = div(df.ebit, df.revenue)
    r["Interpretable"] = (df.ebit > 0) & (uai > 0)
    return r


F5 = ["Carga tributaria", "Carga de intereses", "Margen EBIT", "Rotación de activos", "Multiplicador de capital"]
COLS = [
    "Margen neto", "ROE", "ROA", "Rotación de activos", "Razón corriente",
    "Cobertura de intereses", "Deuda/Patrimonio", "DSO (días)"
]
MENOS_ES_MEJOR = {"Deuda/Patrimonio", "DSO (días)"}
PESOS = {
    "Margen neto": 0.20,
    "ROE": 0.20,
    "ROA": 0.10,
    "Rotación de activos": 0.10,
    "Razón corriente": 0.10,
    "Cobertura de intereses": 0.15,
    "Deuda/Patrimonio": 0.10,
    "DSO (días)": 0.05,
}


def orientar(tabla: pd.DataFrame) -> pd.DataFrame:
    return tabla.apply(lambda s: -s if s.name in MENOS_ES_MEJOR else s)


def percentiles(tabla: pd.DataFrame) -> pd.DataFrame:
    return orientar(tabla).rank(pct=True)


def zscores(tabla: pd.DataFrame, lim=2.5) -> pd.DataFrame:
    x = orientar(tabla)
    sd = x.std(ddof=0).replace(0, np.nan)
    z = (x - x.median()) / sd
    return z.clip(-lim, lim)


def peer_screening(R: pd.DataFrame, anio: int):
    corte = R[R.anio == anio].set_index("empresa")
    X = corte[COLS].copy()
    completos = X.notna().all(axis=1)
    Xc = X.loc[completos]
    pct = percentiles(Xc)
    z = zscores(Xc)
    pesos = pd.Series(PESOS)
    ranking = pd.DataFrame({
        "Puntaje (percentiles)": (pct * pesos).sum(axis=1),
        "Puntaje (z-scores)": (z * pesos).sum(axis=1),
        "Pasa filtro original": (Xc["Cobertura de intereses"] > 3) & (Xc["Razón corriente"] > 0.8),
    })
    ranking = ranking.sort_values(
        ["Puntaje (percentiles)", "Puntaje (z-scores)"], ascending=[False, False]
    )
    ranking.insert(0, "Ranking", range(1, len(ranking) + 1))
    return X, pct, z, ranking


def atribucion(ant, act):
    lr = np.log(act[F5].astype(float) / ant[F5].astype(float))
    if np.isclose(lr.sum(), 0):
        return pd.Series(index=F5, dtype=float)
    return (act["ROE"] - ant["ROE"]) * lr / lr.sum()


def motores(df: pd.DataFrame) -> pd.DataFrame:
    """Mismos motores operacionales del notebook."""
    m = df[["empresa", "anio"]].copy()
    m["Margen bruto"] = div(df.gross_profit, df.revenue)
    m["SG&A / ventas"] = div(df.sga, df.revenue)
    m["Dep. / ventas"] = div(df.depreciation, df.revenue)
    m["Margen EBIT"] = div(df.ebit, df.revenue)
    m["DSO (días)"] = 365 * div(df.receivables, df.revenue)
    m["Inventario (días)"] = 365 * div(df.inventory, df.cogs)
    m["Ventas / PP&E"] = div(df.revenue, df.ppe)
    return m


# =============================================================================
# SEÑALES DE ALERTA
# =============================================================================
def accruals_panel(df: pd.DataFrame) -> pd.DataFrame:
    p = df.set_index(["empresa", "anio"]).sort_index()
    g = p.groupby(level="empresa")
    activos_prom = (p.total_assets + g.total_assets.shift(1)) / 2
    acc_flujo = div(p.net_income - p.cfo, activos_prom)
    acc_balance = div(
        (g.current_assets.diff() - g.cash.diff()) - g.current_liabilities.diff() - p.depreciation,
        activos_prom,
    )
    return pd.DataFrame({"Flujo de caja": acc_flujo, "Balance": acc_balance})


def beneish_indices(t, a):
    tasa_dep = lambda x: x.depreciation / (x.depreciation + x.ppe)
    return pd.Series({
        "DSRI": (t.receivables / t.revenue) / (a.receivables / a.revenue),
        "GMI": (a.gross_profit / a.revenue) / (t.gross_profit / t.revenue),
        "AQI": (1 - (t.current_assets + t.ppe) / t.total_assets) / (1 - (a.current_assets + a.ppe) / a.total_assets),
        "SGI": t.revenue / a.revenue,
        "DEPI": tasa_dep(a) / tasa_dep(t),
        "SGAI": (t.sga / t.revenue) / (a.sga / a.revenue),
        "LVGI": ((t.long_term_debt + t.current_liabilities) / t.total_assets)
                / ((a.long_term_debt + a.current_liabilities) / a.total_assets),
        "TATA": (t.net_income - t.cfo) / t.total_assets,
    }).replace([np.inf, -np.inf], np.nan)


COEF = pd.Series({
    "DSRI": 0.920, "GMI": 0.528, "AQI": 0.404, "SGI": 0.892,
    "DEPI": 0.115, "SGAI": -0.172, "LVGI": -0.327, "TATA": 4.679,
})
CONST = -4.84


def m_score(idx):
    valores = idx.reindex(COEF.index)
    if valores.isna().any() or not np.isfinite(valores.astype(float)).all():
        return np.nan
    return CONST + (COEF * valores).sum()


def beneish_panel(df: pd.DataFrame) -> pd.DataFrame:
    p = df.set_index(["empresa", "anio"]).sort_index()
    filas = []
    for (empresa, anio), actual in p.iterrows():
        if (empresa, anio - 1) in p.index:
            idx = beneish_indices(actual, p.loc[(empresa, anio - 1)])
            filas.append({"empresa": empresa, "anio": anio, **idx, "M-Score": m_score(idx)})
    return pd.DataFrame(filas)


def altman(df: pd.DataFrame) -> pd.DataFrame:
    x1 = div(df.current_assets - df.current_liabilities, df.total_assets)
    x2 = div(df.retained_earnings, df.total_assets)
    x3 = div(df.ebit, df.total_assets)
    x5 = div(df.revenue, df.total_assets)
    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * div(df.market_cap, df.total_liabilities) + 1.0 * x5
    z2 = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * div(df.equity, df.total_liabilities)
    return pd.DataFrame({"Z": z, "Z''": z2}, index=df.index)


def estado_senal(valor, alerta, vigilar, mayor_es_peor=True):
    if pd.isna(valor):
        return "N/D"
    if mayor_es_peor:
        return "Alerta" if valor > alerta else ("Vigilar" if valor > vigilar else "OK")
    return "Alerta" if valor < alerta else ("Vigilar" if valor < vigilar else "OK")


def tablero_alertas(panel: pd.DataFrame, anio: int) -> pd.DataFrame:
    empresas = sorted(panel.empresa.unique())
    accr = accruals_panel(panel)["Flujo de caja"].unstack("empresa")
    delta = pd.Series(index=empresas, dtype=float)
    if anio in accr.index and anio - 1 in accr.index:
        delta = (accr.loc[anio] - accr.loc[anio - 1]).reindex(empresas) * 100

    ben = beneish_panel(panel)
    if not ben.empty and anio in ben.anio.values:
        m = ben[ben.anio == anio].set_index("empresa")["M-Score"].reindex(empresas)
    else:
        m = pd.Series(index=empresas, dtype=float)

    p_anio = panel[panel.anio == anio].set_index("empresa").reindex(empresas)
    zvals = altman(p_anio)

    out = pd.DataFrame({
        "Δ accruals (pp)": delta,
        "M-Score": m,
        "Z": zvals["Z"],
        "Z''": zvals["Z''"],
    }, index=empresas)
    out["Accruals"] = out["Δ accruals (pp)"].apply(estado_senal, args=(2.0, 1.5))
    out["Beneish"] = out["M-Score"].apply(estado_senal, args=(-1.78, -2.22))
    out["Altman Z''"] = out["Z''"].apply(estado_senal, args=(1.10, 2.60, False))
    return out


# =============================================================================
# COMMON-SIZE Y HELPERS
# =============================================================================
def common_size_resultados(df_anio: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Costo de ventas": df_anio.cogs,
        "Gastos de adm. y ventas": df_anio.sga,
        "Depreciación": df_anio.depreciation,
        "Resultado operacional": df_anio.ebit,
    }, index=df_anio.index).div(df_anio.revenue, axis=0)


def common_size_balance(df_anio: pd.DataFrame):
    comp = df_anio[["cash", "receivables", "inventory", "ppe"]].fillna(0)
    otros_ac = df_anio.current_assets - comp.cash - comp.receivables - comp.inventory
    otros_anc = df_anio.total_assets - df_anio.current_assets - comp.ppe
    activos = pd.DataFrame({
        "Caja": comp.cash,
        "Cuentas por cobrar": comp.receivables,
        "Inventario": comp.inventory,
        "Otros AC": otros_ac,
        "PP&E": comp.ppe,
        "Otros ANC": otros_anc,
    }, index=df_anio.index)
    pasivos = pd.DataFrame({
        "Pasivo corriente": df_anio.current_liabilities,
        "Deuda LP y otros": df_anio.total_liabilities - df_anio.current_liabilities,
        "Patrimonio": df_anio.equity,
        "Minoritarios y otros": df_anio.total_assets - df_anio.total_liabilities - df_anio.equity,
    }, index=df_anio.index)
    return activos.div(df_anio.total_assets, axis=0), pasivos.div(df_anio.total_assets, axis=0)


def cagr(serie: pd.Series):
    serie = serie.dropna()
    if len(serie) < 2 or serie.iloc[0] <= 0 or serie.iloc[-1] <= 0:
        return np.nan
    T = int(serie.index[-1] - serie.index[0])
    if T <= 0:
        return np.nan
    return (serie.iloc[-1] / serie.iloc[0]) ** (1 / T) - 1


def fmt_ratio(nombre, valor):
    if pd.isna(valor):
        return "N/D"
    if nombre in {"Margen bruto", "Margen operacional", "Margen neto", "ROA", "ROE", "SG&A / ventas", "Dep. / ventas", "Margen EBIT"}:
        return f"{valor:.1%}"
    if nombre in {"DSO (días)", "Inventario (días)", "Días de inventario"}:
        return f"{valor:.1f} días"
    if nombre in {"Rotación de activos", "Razón corriente", "Prueba ácida", "Deuda/Patrimonio", "Cobertura de intereses", "Ventas/PP&E", "Ventas / PP&E", "Multiplicador de capital"}:
        return f"{valor:.2f}x"
    return f"{valor:.2f}"


def plot_layout(fig, height=360, legend=True):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=55, b=25),
        font=dict(family="Arial, sans-serif", color=DARK),
        title_font=dict(color=NAVY, size=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) if legend else None,
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    return fig


def status_badge(label):
    color = {"OK": GREEN, "Vigilar": AMBER, "Alerta": RED, "N/D": GREY}.get(label, GREY)
    return f"<span style='display:inline-block;padding:.18rem .55rem;border-radius:999px;background:{color}18;color:{color};font-weight:700;font-size:.78rem'>{label}</span>"


def relative_strengths(pct: pd.DataFrame, empresa: str):
    if empresa not in pct.index:
        return [], []
    s = pct.loc[empresa].sort_values(ascending=False)
    return list(s.head(3).index), list(s.tail(2).index)


def balance_check(df: pd.DataFrame):
    descuadre = (df.total_assets - df.total_liabilities - df.equity) / df.total_assets
    return descuadre.abs().max()


# =============================================================================
# CARGA Y CONTROLES
# =============================================================================
st.sidebar.markdown("### ✈️ Taller 2")
st.sidebar.caption("Screener financiero · Finanzas Computacionales")
subido = st.sidebar.file_uploader("Actualizar panel (CSV)", type="csv", help="Opcional. Si no cargas nada, se usa panel_para_dashboard.csv del repositorio.")
panel = cargar(subido)

if panel.empty:
    st.title("Taller 2 — Screener financiero de aerolíneas")
    st.info("No se encontró `panel_para_dashboard.csv`. Súbelo desde la barra lateral.")
    st.stop()

columnas_min = {
    "empresa", "anio", "revenue", "cogs", "gross_profit", "sga", "depreciation", "ebit",
    "interest_expense", "net_income", "total_assets", "current_assets", "cash", "receivables",
    "inventory", "ppe", "total_liabilities", "current_liabilities", "long_term_debt", "equity",
    "retained_earnings", "cfo", "market_cap"
}
faltan = sorted(columnas_min - set(panel.columns))
if faltan:
    st.error("Faltan columnas requeridas: " + ", ".join(faltan))
    st.stop()

R = ratios(panel)
M4 = motores(panel)
empresas = sorted(R.empresa.dropna().unique())
anios = sorted(int(x) for x in R.anio.dropna().unique())

foco_default = empresas.index("DAL") if "DAL" in empresas else 0
foco = st.sidebar.selectbox("Empresa foco", empresas, index=foco_default)
anio = st.sidebar.select_slider("Año de análisis", options=anios, value=anios[-1])
pares_disp = [e for e in empresas if e != foco]
pares = st.sidebar.multiselect("Pares", pares_disp, default=pares_disp)
if not pares:
    st.sidebar.warning("Selecciona al menos un par.")
    st.stop()

st.sidebar.divider()
st.sidebar.markdown("**Metodología del ranking**")
st.sidebar.caption("Percentiles ponderados como criterio principal. Z-score como desempate.")
st.sidebar.markdown("**Fuente**")
st.sidebar.caption("Estados financieros del panel utilizado en el Colab.")

fila_df = R[(R.empresa == foco) & (R.anio == anio)]
if fila_df.empty:
    st.error(f"No hay datos para {foco} en {anio}.")
    st.stop()
fila = fila_df.iloc[0]
previo = R[(R.empresa == foco) & (R.anio == anio - 1)]
med_pares = R[(R.empresa.isin(pares)) & (R.anio == anio)].median(numeric_only=True)

R_sel = R[R.empresa.isin([foco] + pares)]
panel_sel = panel[panel.empresa.isin([foco] + pares)]
X, pct, z, ranking = peer_screening(R_sel, anio)
alertas = tablero_alertas(panel_sel, anio)

# =============================================================================
# CABECERA EJECUTIVA
# =============================================================================
ganador = ranking.index[0] if len(ranking) else None
pos_foco = int(ranking.loc[foco, "Ranking"]) if foco in ranking.index else None
score_foco = ranking.loc[foco, "Puntaje (percentiles)"] if foco in ranking.index else np.nan
fortalezas, debilidades = relative_strengths(pct, foco)

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-kicker">Taller 2 · Análisis financiero computacional</div>
        <div class="hero-title">Screener financiero de aerolíneas</div>
        <p class="hero-copy"><b>{anio}</b> · Empresa foco: <b>{foco}</b> · Pares: {', '.join(pares)}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# KPIs cabecera
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Ranking", f"#{pos_foco}" if pos_foco else "N/D")
k2.metric("Puntaje", f"{score_foco:.2f}" if pd.notna(score_foco) else "N/D")
k3.metric("ROE", fmt_ratio("ROE", fila["ROE"]))
k4.metric("Cobertura intereses", fmt_ratio("Cobertura de intereses", fila["Cobertura de intereses"]))
k5.metric("Razón corriente", fmt_ratio("Razón corriente", fila["Razón corriente"]))

# Calidad de datos visible pero discreta
max_desc = balance_check(panel_sel)
periodos = panel_sel.groupby("empresa").anio.nunique().min()
st.caption(f"Control de calidad: {len(set([foco] + pares))} empresas · mínimo {periodos} años por empresa · descuadre máximo del balance {max_desc:.2%}.")

# =============================================================================
# NAVEGACIÓN
# =============================================================================
tab_exec, tab_dupont, tab_motores, tab_tend, tab_riesgo = st.tabs([
    "Resumen ejecutivo",
    "DuPont",
    "Motores operacionales",
    "Tendencias & common-size",
    "Riesgos & alertas",
])

# =============================================================================
# 1) RESUMEN EJECUTIVO
# =============================================================================
with tab_exec:
    c_left, c_right = st.columns([1.25, 1])

    with c_left:
        st.subheader("Ranking final")
        # Convertir el índice del ranking en una columna de empresa de forma robusta
        rplot = ranking.reset_index()
        rplot = rplot.rename(columns={rplot.columns[0]: "Empresa"})
        rplot = rplot.sort_values("Ranking", ascending=False).copy()
        rplot["Tipo"] = np.where(rplot["Empresa"].eq(foco), foco, "Pares")

        fig_rank = px.bar(
            rplot,
            x="Puntaje (percentiles)",
            y="Empresa",
            orientation="h",
            text="Puntaje (percentiles)",
            color="Tipo",
            color_discrete_map={foco: ACCENT, "Pares": NAVY},
            title="Percentiles ponderados: mayor puntaje = mejor posición relativa",
        )
        fig_rank.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_rank.update_xaxes(range=[0, 1.08], title=None)
        fig_rank.update_yaxes(title=None)
        fig_rank.update_layout(showlegend=False)
        st.plotly_chart(plot_layout(fig_rank, 335, legend=False), use_container_width=True, config={"displaylogo": False})

    with c_right:
        st.subheader("Lectura ejecutiva")
        if ganador:
            st.markdown(
                f"""
                <div class="section-note">
                El screener ubica a <b>{ganador}</b> en el primer lugar para {anio}. El ranking combina
                rentabilidad, eficiencia, liquidez y endeudamiento; las señales de alerta se analizan por separado.
                </div>
                """,
                unsafe_allow_html=True,
            )
        if fortalezas:
            st.markdown("**Fortalezas relativas de la empresa foco**")
            st.markdown(" ".join([f"<span class='pill'>{x}</span>" for x in fortalezas]), unsafe_allow_html=True)
        if debilidades:
            st.markdown("**Debilidades relativas a monitorear**")
            st.markdown(" ".join([f"<span class='pill'>{x}</span>" for x in debilidades]), unsafe_allow_html=True)

        if foco in alertas.index:
            st.markdown("**Semáforo de riesgo**")
            ar = alertas.loc[foco]
            st.markdown(
                f"Accruals {status_badge(ar['Accruals'])} &nbsp; "
                f"Beneish {status_badge(ar['Beneish'])} &nbsp; "
                "Altman Z'' " + status_badge(ar["Altman Z''"]),
                unsafe_allow_html=True,
            )

    st.subheader("Peer screening")
    if not pct.empty:
        Xc = X.loc[pct.index, COLS]
        texto = [[fmt_ratio(c, Xc.loc[e, c]) for c in COLS] for e in pct.index]
        heat = go.Figure(go.Heatmap(
            z=pct[COLS].values,
            x=COLS,
            y=pct.index,
            text=texto,
            texttemplate="%{text}",
            colorscale=[[0, ACCENT], [0.5, "#F4E4D8"], [0.75, "#DDE7F3"], [1, NAVY]],
            zmin=0, zmax=1,
            showscale=False,
            hovertemplate="%{y}<br>%{x}<br>Percentil: %{z:.0%}<extra></extra>",
        ))
        heat.update_layout(title="Azul = mejor posición relativa · naranja = peor posición relativa")
        heat.update_xaxes(tickangle=-20)
        st.plotly_chart(plot_layout(heat, 355, legend=False), use_container_width=True, config={"displaylogo": False})

    q1, q2 = st.columns([1.2, 1])
    with q1:
        st.subheader("Ranking detallado")
        rshow = ranking.copy()
        st.dataframe(
            rshow.style.format({"Puntaje (percentiles)": "{:.2f}", "Puntaje (z-scores)": "{:+.2f}"}),
            use_container_width=True,
        )
    with q2:
        st.subheader("Metodología")
        pesos_df = pd.DataFrame({
            "Peso": pd.Series(PESOS),
            "Orientación": ["Menor es mejor" if c in MENOS_ES_MEJOR else "Mayor es mejor" for c in PESOS],
        })
        st.dataframe(pesos_df.style.format({"Peso": "{:.0%}"}), use_container_width=True)

    if not ranking["Pasa filtro original"].any():
        st.info(
            "El filtro absoluto del ejemplo original (cobertura > 3x y razón corriente > 0,8x) no discrimina este grupo: "
            "ninguna empresa cumple ambos criterios. La decisión final se basa en la comparación relativa entre pares."
        )

    st.subheader("Recomendación")
    if ganador is not None:
        winner_row = R[(R.empresa == ganador) & (R.anio == anio)].iloc[0]
        winner_risk = alertas.loc[ganador] if ganador in alertas.index else pd.Series(dtype=object)
        st.markdown(
            f"""
            <div class="section-note">
            <b>{ganador}</b> es la alternativa preferida del screener en {anio}. Presenta un ROE de
            <b>{fmt_ratio('ROE', winner_row['ROE'])}</b>, margen neto de
            <b>{fmt_ratio('Margen neto', winner_row['Margen neto'])}</b> y cobertura de intereses de
            <b>{fmt_ratio('Cobertura de intereses', winner_row['Cobertura de intereses'])}</b>.
            La recomendación se complementa con el análisis de liquidez y con las señales de riesgo del tablero.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Ver ratios de las cuatro familias"):
        familias = {
            "Liquidez": ["Razón corriente", "Prueba ácida"],
            "Rentabilidad": ["Margen bruto", "Margen operacional", "Margen neto", "ROA", "ROE"],
            "Endeudamiento": ["Deuda/Patrimonio", "Cobertura de intereses"],
            "Eficiencia": ["Rotación de activos", "DSO (días)", "Días de inventario"],
        }
        cols_f = st.columns(4)
        for col_f, (familia, indicadores) in zip(cols_f, familias.items()):
            col_f.markdown(f"**{familia}**")
            for ind in indicadores:
                col_f.write(f"{ind}: **{fmt_ratio(ind, fila[ind])}**")

# =============================================================================
# 2) DUPONT
# =============================================================================
with tab_dupont:
    st.subheader("DuPont de 3 factores")
    d1, d2 = st.columns(2)
    comp = ["Margen neto", "Rotación de activos", "Multiplicador de capital", "ROE"]
    comp_pct = {"Margen neto", "ROE"}
    rel = pd.DataFrame({
        "Componente": comp,
        "Relativo": [div(fila[c], med_pares[c]) for c in comp],
        "Etiqueta": [
            f"{fila[c]:.1%} vs {med_pares[c]:.1%}" if c in comp_pct else f"{fila[c]:.2f} vs {med_pares[c]:.2f}"
            for c in comp
        ],
    })
    fig_d3 = px.bar(
        rel, x="Componente", y="Relativo", text="Etiqueta",
        color=rel["Componente"].eq("ROE").map({True: "ROE", False: "Factor"}),
        color_discrete_map={"ROE": ACCENT, "Factor": NAVY},
        title=f"{foco}: componentes del ROE frente a la mediana de pares",
    )
    fig_d3.add_hline(y=1, line_dash="dash", line_color=GREY)
    fig_d3.update_traces(textposition="outside")
    fig_d3.update_layout(showlegend=False)
    fig_d3.update_yaxes(title="Veces la mediana")
    d1.plotly_chart(plot_layout(fig_d3, 350, legend=False), use_container_width=True, config={"displaylogo": False})

    hist = R[R.empresa.isin([foco] + pares)].copy()
    hist["Serie"] = hist.empresa.where(hist.empresa == foco, "Mediana pares")
    roe_t = hist.groupby(["anio", "Serie"], as_index=False)["ROE"].median()
    fig_roe = px.line(
        roe_t, x="anio", y="ROE", color="Serie", markers=True,
        color_discrete_map={foco: ACCENT, "Mediana pares": GREY},
        title=f"ROE en el tiempo: {foco} vs. mediana",
    )
    fig_roe.update_yaxes(tickformat=".0%")
    fig_roe.update_xaxes(dtick=1, title=None)
    d2.plotly_chart(plot_layout(fig_roe, 350), use_container_width=True, config={"displaylogo": False})

    st.subheader("DuPont de 5 factores y atribución del ROE")
    d3, d4 = st.columns([1, 1.35])
    t5 = pd.DataFrame({
        "Factor": F5,
        foco: [fila[c] for c in F5],
        "Mediana pares": [med_pares[c] for c in F5],
    })
    t5["Relativo"] = t5[foco] / t5["Mediana pares"]
    d3.dataframe(
        t5.set_index("Factor").style.format({foco: "{:.3f}", "Mediana pares": "{:.3f}", "Relativo": "{:.2f}x"}),
        use_container_width=True,
    )
    if not bool(fila["Interpretable"]):
        d3.warning("EBIT o utilidad antes de impuestos negativos: las cargas no son interpretables.")

    if previo.empty:
        d4.info("Se necesita el año anterior para atribuir la variación del ROE.")
    elif (fila[F5] <= 0).any() or (previo.iloc[0][F5] <= 0).any():
        d4.warning("La atribución logarítmica exige factores positivos en ambos años.")
    else:
        ant = previo.iloc[0]
        aporte = atribucion(ant, fila)
        mayor = aporte.abs().idxmax()
        wf = go.Figure(go.Waterfall(
            x=[f"ROE<br>{anio-1}", "Carga<br>tributaria", "Carga<br>intereses", "Margen<br>EBIT", "Rotación", "Multiplicador", f"ROE<br>{anio}"],
            measure=["absolute"] + ["relative"] * len(F5) + ["total"],
            y=[ant["ROE"]] + list(aporte.values) + [fila["ROE"]],
            text=[f"{ant['ROE']:.1%}"] + [f"{v*100:+.1f} pp" for v in aporte.values] + [f"{fila['ROE']:.1%}"],
            textposition="outside",
            connector={"line": {"color": GREY, "dash": "dot"}},
            increasing={"marker": {"color": GREEN}},
            decreasing={"marker": {"color": RED}},
            totals={"marker": {"color": ACCENT}},
        ))
        wf.update_layout(title=f"El mayor aporte al cambio del ROE proviene de {mayor}", yaxis_tickformat=".0%")
        d4.plotly_chart(plot_layout(wf, 370, legend=False), use_container_width=True, config={"displaylogo": False})

    st.markdown(
        "<div class='section-note'>DuPont separa si el ROE proviene de márgenes, eficiencia de activos o apalancamiento. "
        "La atribución cuantifica qué factor explica el cambio interanual.</div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# 3) MOTORES OPERACIONALES
# =============================================================================
with tab_motores:
    st.subheader("Motores operacionales")
    m4 = M4[(M4.anio == anio) & (M4.empresa.isin([foco] + pares))].set_index("empresa").drop(columns="anio")
    med_m4 = m4.loc[pares].median(numeric_only=True)
    foco_m4 = m4.loc[foco]

    residuo = foco_m4["Margen bruto"] - foco_m4["SG&A / ventas"] - foco_m4["Dep. / ventas"] - foco_m4["Margen EBIT"]

    a, b, c, d = st.columns(4)
    a.metric("Margen bruto", fmt_ratio("Margen bruto", foco_m4["Margen bruto"]), f"{(foco_m4['Margen bruto']-med_m4['Margen bruto'])*100:+.1f} pp vs mediana")
    b.metric("Margen EBIT", fmt_ratio("Margen EBIT", foco_m4["Margen EBIT"]), f"{(foco_m4['Margen EBIT']-med_m4['Margen EBIT'])*100:+.1f} pp vs mediana")
    c.metric("DSO", fmt_ratio("DSO (días)", foco_m4["DSO (días)"]), f"{foco_m4['DSO (días)']-med_m4['DSO (días)']:+.1f} días vs mediana", delta_color="inverse")
    d.metric("Ventas / PP&E", fmt_ratio("Ventas / PP&E", foco_m4["Ventas / PP&E"]), f"{foco_m4['Ventas / PP&E']-med_m4['Ventas / PP&E']:+.2f}x vs mediana")

    op1, op2 = st.columns(2)
    bridge = pd.DataFrame({
        "Componente": ["Margen bruto", "SG&A / ventas", "Dep. / ventas", "Margen EBIT"],
        foco: [foco_m4["Margen bruto"], foco_m4["SG&A / ventas"], foco_m4["Dep. / ventas"], foco_m4["Margen EBIT"]],
        "Mediana pares": [med_m4["Margen bruto"], med_m4["SG&A / ventas"], med_m4["Dep. / ventas"], med_m4["Margen EBIT"]],
    })
    bridge_long = bridge.melt(id_vars="Componente", var_name="Serie", value_name="Valor")
    fig_bridge = px.bar(
        bridge_long, x="Componente", y="Valor", color="Serie", barmode="group",
        color_discrete_map={foco: ACCENT, "Mediana pares": NAVY},
        title=f"Estructura de margen: {foco} vs. mediana de pares",
    )
    fig_bridge.update_yaxes(tickformat=".0%", title=None)
    op1.plotly_chart(plot_layout(fig_bridge, 360), use_container_width=True, config={"displaylogo": False})

    eff_cols = ["DSO (días)", "Inventario (días)", "Ventas / PP&E"]
    eff = pd.DataFrame({
        "Indicador": eff_cols,
        foco: [foco_m4[x] for x in eff_cols],
        "Mediana pares": [med_m4[x] for x in eff_cols],
    })
    eff["Empresa foco"] = [fmt_ratio(x, foco_m4[x]) for x in eff_cols]
    eff["Mediana"] = [fmt_ratio(x, med_m4[x]) for x in eff_cols]
    op2.markdown("#### Eficiencia operacional")
    op2.dataframe(eff[["Indicador", "Empresa foco", "Mediana"]].set_index("Indicador"), use_container_width=True)
    op2.markdown(
        f"<div class='section-note'>El puente simplificado del margen deja un residuo de <b>{residuo:.1%}</b> para {foco}. "
        "Esto refleja costos operacionales no capturados por SG&A y depreciación, especialmente relevantes en aerolíneas.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Comparación completa de motores")
    tabla_m = pd.concat([m4, med_m4.to_frame("Mediana").T])
    fmts = {
        "Margen bruto": "{:.1%}", "SG&A / ventas": "{:.1%}", "Dep. / ventas": "{:.1%}", "Margen EBIT": "{:.1%}",
        "DSO (días)": "{:.1f}", "Inventario (días)": "{:.1f}", "Ventas / PP&E": "{:.2f}",
    }
    st.dataframe(tabla_m.style.format(fmts), use_container_width=True)

# =============================================================================
# 4) TENDENCIAS & COMMON-SIZE
# =============================================================================
with tab_tend:
    st.subheader("Tendencias")
    ventas = panel_sel.pivot(index="anio", columns="empresa", values="revenue").sort_index()
    indice = 100 * ventas / ventas.iloc[0]
    indice_long = indice.reset_index().melt(id_vars="anio", var_name="empresa", value_name="Índice")
    cmap = {e: (ACCENT if e == foco else GREY) for e in [foco] + pares}

    t1, t2 = st.columns(2)
    fig_idx = px.line(
        indice_long, x="anio", y="Índice", color="empresa", markers=True,
        color_discrete_map=cmap,
        title=f"Ventas: índice base {int(indice.index[0])} = 100",
    )
    fig_idx.add_hline(y=100, line_dash="dot", line_color=LIGHT)
    fig_idx.update_xaxes(dtick=1, title=None)
    t1.plotly_chart(plot_layout(fig_idx, 365), use_container_width=True, config={"displaylogo": False})

    mop = R_sel.pivot(index="anio", columns="empresa", values="Margen operacional").sort_index()
    med = mop[pares].median(axis=1)
    minimo = mop[pares].min(axis=1)
    maximo = mop[pares].max(axis=1)
    fig_mop = go.Figure()
    fig_mop.add_trace(go.Scatter(x=mop.index, y=maximo, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig_mop.add_trace(go.Scatter(x=mop.index, y=minimo, fill="tonexty", fillcolor="rgba(46,84,150,0.10)", line=dict(width=0), name="Rango pares", hoverinfo="skip"))
    fig_mop.add_trace(go.Scatter(x=mop.index, y=med, mode="lines+markers", name="Mediana pares", line=dict(color=NAVY, dash="dash")))
    fig_mop.add_trace(go.Scatter(x=mop.index, y=mop[foco], mode="lines+markers", name=foco, line=dict(color=ACCENT, width=3)))
    fig_mop.update_layout(title="Margen operacional: foco vs. rango de pares")
    fig_mop.update_yaxes(tickformat=".0%")
    fig_mop.update_xaxes(dtick=1, title=None)
    t2.plotly_chart(plot_layout(fig_mop, 365), use_container_width=True, config={"displaylogo": False})

    crec = ventas.apply(cagr).sort_values(ascending=False)
    st.subheader("CAGR de ventas")
    cagr_df = crec.rename("CAGR").reset_index()
    fig_cagr = px.bar(
        cagr_df, x="empresa", y="CAGR", text=cagr_df.CAGR.map(lambda x: "N/D" if pd.isna(x) else f"{x:.1%}"),
        color=cagr_df.empresa.eq(foco).map({True: foco, False: "Pares"}),
        color_discrete_map={foco: ACCENT, "Pares": NAVY},
        title="Crecimiento anual compuesto del período disponible",
    )
    fig_cagr.update_yaxes(tickformat=".0%", title=None)
    fig_cagr.update_layout(showlegend=False)
    st.plotly_chart(plot_layout(fig_cagr, 320, legend=False), use_container_width=True, config={"displaylogo": False})

    st.subheader(f"Common-size {anio}")
    ult = panel_sel[panel_sel.anio == anio].set_index("empresa")
    cs_er = common_size_resultados(ult)
    cs_act, cs_pas = common_size_balance(ult)

    cs1, cs2 = st.columns(2)
    er_cmp = pd.DataFrame({foco: cs_er.loc[foco], "Mediana pares": cs_er.loc[pares].median()})
    cs1.markdown("#### Estado de resultados (% de ventas)")
    cs1.dataframe(er_cmp.style.format("{:.1%}"), use_container_width=True)

    balance_focus = pd.DataFrame({
        "Activos": cs_act.loc[foco],
    })
    fig_act = px.bar(
        balance_focus.reset_index(), x="Activos", y="index", orientation="h",
        text="Activos", title=f"{foco}: composición de activos",
        color_discrete_sequence=[NAVY],
    )
    fig_act.update_traces(texttemplate="%{text:.1%}", textposition="inside")
    fig_act.update_xaxes(tickformat=".0%", title=None, range=[0, 1])
    fig_act.update_yaxes(title=None, categoryorder="total ascending")
    cs2.plotly_chart(plot_layout(fig_act, 335, legend=False), use_container_width=True, config={"displaylogo": False})

    with st.expander("Ver common-size completo del balance"):
        b1, b2 = st.columns(2)
        b1.markdown("**Activos (% de activos totales)**")
        b1.dataframe(cs_act.style.format("{:.1%}"), use_container_width=True)
        b2.markdown("**Pasivos y patrimonio (% de activos totales)**")
        b2.dataframe(cs_pas.style.format("{:.1%}"), use_container_width=True)

# =============================================================================
# 5) RIESGOS & ALERTAS
# =============================================================================
with tab_riesgo:
    st.subheader("Señales de alerta")
    af = alertas.loc[foco] if foco in alertas.index else pd.Series(dtype=float)

    r1, r2, r3 = st.columns(3)
    r1.metric("Δ accruals", "N/D" if pd.isna(af.get("Δ accruals (pp)")) else f"{af['Δ accruals (pp)']:+.1f} pp")
    r1.markdown(status_badge(af.get("Accruals", "N/D")), unsafe_allow_html=True)
    r2.metric("Beneish M-Score", "N/D" if pd.isna(af.get("M-Score")) else f"{af['M-Score']:.2f}")
    r2.markdown(status_badge(af.get("Beneish", "N/D")), unsafe_allow_html=True)
    r3.metric("Altman Z''", "N/D" if pd.isna(af.get("Z''")) else "{:.2f}".format(af["Z''"]))
    r3.markdown(status_badge(af.get("Altman Z''", "N/D")), unsafe_allow_html=True)

    st.caption("Beneish se muestra como N/D si falta cualquiera de sus ocho componentes; no se calcula un score parcial.")

    # Tablero grupo
    orden_est = ["Accruals", "Beneish", "Altman Z''"]
    zmap = {"Alerta": 0, "Vigilar": 0.5, "OK": 1, "N/D": 0.25}
    zvals = alertas[orden_est].applymap(zmap.get)
    textos = []
    for e in alertas.index:
        textos.append([
            f"{alertas.loc[e, 'Accruals']}<br>{alertas.loc[e, 'Δ accruals (pp)']:+.1f} pp" if pd.notna(alertas.loc[e, "Δ accruals (pp)"]) else "N/D",
            f"{alertas.loc[e, 'Beneish']}<br>M={alertas.loc[e, 'M-Score']:.2f}" if pd.notna(alertas.loc[e, "M-Score"]) else "N/D",
            ("{}<br>Z''={:.2f}".format(alertas.loc[e, "Altman Z''"], alertas.loc[e, "Z''"]) if pd.notna(alertas.loc[e, "Z''"]) else "N/D"),
        ])
    fig_alert = go.Figure(go.Heatmap(
        z=zvals.values, x=orden_est, y=alertas.index,
        text=textos, texttemplate="%{text}",
        colorscale=[[0, RED], [0.49, RED], [0.5, AMBER], [0.74, AMBER], [0.75, GREEN], [1, GREEN]],
        zmin=0, zmax=1, showscale=False, hoverinfo="skip",
    ))
    fig_alert.update_layout(title="Tablero de alertas del grupo")
    st.plotly_chart(plot_layout(fig_alert, 330, legend=False), use_container_width=True, config={"displaylogo": False})

    rr1, rr2 = st.columns(2)
    with rr1:
        st.subheader("Accruals")
        accr = accruals_panel(panel_sel)
        acc_f = accr["Flujo de caja"].unstack("empresa")
        acc_long = acc_f.reset_index().melt(id_vars="anio", var_name="empresa", value_name="Accruals")
        fig_acc = px.line(
            acc_long, x="anio", y="Accruals", color="empresa", markers=True,
            color_discrete_map={e: (ACCENT if e == foco else GREY) for e in [foco] + pares},
            title="Ratio de accruals basado en flujo de caja",
        )
        fig_acc.update_yaxes(tickformat=".0%")
        fig_acc.update_xaxes(dtick=1, title=None)
        st.plotly_chart(plot_layout(fig_acc, 335), use_container_width=True, config={"displaylogo": False})

    with rr2:
        st.subheader("Altman")
        az = altman(panel_sel[panel_sel.anio == anio].set_index("empresa"))
        az_long = az.reset_index().melt(id_vars="empresa", var_name="Modelo", value_name="Score")
        fig_az = px.bar(
            az_long, x="empresa", y="Score", color="Modelo", barmode="group",
            color_discrete_map={"Z": NAVY, "Z''": ACCENT},
            title="Altman Z y Z''",
        )
        fig_az.add_hline(y=1.10, line_dash="dash", line_color=RED, annotation_text="Z'' 1,10")
        fig_az.add_hline(y=2.60, line_dash="dot", line_color=GREEN, annotation_text="Z'' 2,60")
        st.plotly_chart(plot_layout(fig_az, 335), use_container_width=True, config={"displaylogo": False})

    with st.expander("Interpretación y umbrales"):
        st.markdown(
            "- **Accruals:** Alerta si el salto supera +2,0 pp; Vigilar sobre +1,5 pp.\n"
            "- **Beneish M-Score:** Alerta sobre −1,78; Vigilar sobre −2,22. Señal de revisión, no prueba de manipulación.\n"
            "- **Altman Z'':** Alerta bajo 1,10; zona gris entre 1,10 y 2,60; sobre 2,60 zona segura.\n"
            "- **Altman Z:** se muestra como referencia adicional; para este grupo, la señal principal del Taller 2 es Z'' para no manufactureras."
        )

# =============================================================================
# PIE DE PÁGINA
# =============================================================================
st.divider()
st.caption(
    "Taller 2 · Finanzas Computacionales · Magíster en Business Analytics (UAI) · "
    "Dashboard construido sobre el mismo panel, fórmulas, pesos y umbrales utilizados en el Colab final."
)
