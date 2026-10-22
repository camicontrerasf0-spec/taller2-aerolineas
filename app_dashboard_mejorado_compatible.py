"""Dashboard Taller 2 — Screener financiero de aerolíneas.

Ejecución:
    pip install streamlit plotly pandas numpy
    streamlit run app_dashboard_mejorado.py

El dashboard usa un panel en formato largo (una fila por empresa-año). Puede cargar
`panel_para_dashboard.csv` desde la barra lateral o usarlo automáticamente si está en
la misma carpeta del script.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------- estilo
NAVY = "#1F3864"
BLUE = "#2E5496"
ACCENT = "#C55A11"
GREY = "#8A8F98"
LIGHT = "#DEE7F2"
GREEN = "#2E7D57"
RED = "#B23A3A"
AMBER = "#C9A227"

st.set_page_config(page_title="Taller 2 | Screener de aerolíneas", layout="wide")

# ---------------------------------------------------------------- datos
@st.cache_data
def cargar(archivo) -> pd.DataFrame:
    """Carga el panel subido o busca panel_para_dashboard.csv junto al script."""
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
    """División segura: reemplaza infinitos por NaN."""
    out = a / b
    if isinstance(out, pd.Series):
        return out.replace([np.inf, -np.inf], np.nan)
    return np.nan if not np.isfinite(out) else out


def ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Cuatro familias de ratios + componentes DuPont por empresa-año."""
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

    # DuPont de 3 y 5 factores
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
COLS = ["Margen neto", "ROE", "ROA", "Rotación de activos", "Razón corriente",
        "Cobertura de intereses", "Deuda/Patrimonio", "DSO (días)"]
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
        "Filtro profesor": (Xc["Cobertura de intereses"] > 3) & (Xc["Razón corriente"] > 0.8),
    })
    # Percentiles como criterio principal; z-score como desempate.
    ranking = ranking.sort_values(["Puntaje (percentiles)", "Puntaje (z-scores)"], ascending=False)
    ranking.insert(0, "Ranking", range(1, len(ranking) + 1))
    return X, pct, z, ranking


def atribucion(ant, act):
    lr = np.log(act[F5].astype(float) / ant[F5].astype(float))
    if np.isclose(lr.sum(), 0):
        return pd.Series(index=F5, dtype=float)
    return (act["ROE"] - ant["ROE"]) * lr / lr.sum()


# ---------------------------------------------------------------- señales de alerta
def accruals_flujo(df: pd.DataFrame) -> pd.Series:
    p = df.set_index(["empresa", "anio"]).sort_index()
    g = p.groupby(level="empresa")
    activos_prom = (p.total_assets + g.total_assets.shift(1)) / 2
    return ((p.net_income - p.cfo) / activos_prom).replace([np.inf, -np.inf], np.nan)


def beneish_indices(t, a):
    tasa_dep = lambda x: x.depreciation / (x.depreciation + x.ppe)
    return pd.Series({
        "DSRI": (t.receivables / t.revenue) / (a.receivables / a.revenue),
        "GMI":  (a.gross_profit / a.revenue) / (t.gross_profit / t.revenue),
        "AQI":  (1 - (t.current_assets + t.ppe) / t.total_assets) / (1 - (a.current_assets + a.ppe) / a.total_assets),
        "SGI":  t.revenue / a.revenue,
        "DEPI": tasa_dep(a) / tasa_dep(t),
        "SGAI": (t.sga / t.revenue) / (a.sga / a.revenue),
        "LVGI": ((t.long_term_debt + t.current_liabilities) / t.total_assets) /
                ((a.long_term_debt + a.current_liabilities) / a.total_assets),
        "TATA": (t.net_income - t.cfo) / t.total_assets,
    }).replace([np.inf, -np.inf], np.nan)


COEF = pd.Series({"DSRI": 0.920, "GMI": 0.528, "AQI": 0.404, "SGI": 0.892,
                  "DEPI": 0.115, "SGAI": -0.172, "LVGI": -0.327, "TATA": 4.679})
CONST = -4.84


def m_score(idx):
    vals = idx.reindex(COEF.index)
    # Mejora frente al ejemplo: no calcular un M-Score parcial si falta un índice.
    if vals.isna().any() or not np.isfinite(vals.astype(float)).all():
        return np.nan
    return CONST + (COEF * vals).sum()


def beneish_panel(df: pd.DataFrame) -> pd.DataFrame:
    p = df.set_index(["empresa", "anio"]).sort_index()
    filas = []
    for (e, an), t in p.iterrows():
        if (e, an - 1) in p.index:
            idx = beneish_indices(t, p.loc[(e, an - 1)])
            filas.append({"empresa": e, "anio": an, **idx, "M-Score": m_score(idx)})
    return pd.DataFrame(filas)


def altman_z2(df: pd.DataFrame) -> pd.Series:
    x1 = div(df.current_assets - df.current_liabilities, df.total_assets)
    x2 = div(df.retained_earnings, df.total_assets)
    x3 = div(df.ebit, df.total_assets)
    x4 = div(df.equity, df.total_liabilities)
    return 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4


def estado_senal(valor, alerta, vigilar, mayor_es_peor=True):
    if pd.isna(valor):
        return "N/D"
    if mayor_es_peor:
        return "Alerta" if valor > alerta else ("Vigilar" if valor > vigilar else "OK")
    return "Alerta" if valor < alerta else ("Vigilar" if valor < vigilar else "OK")


def tablero_alertas(panel: pd.DataFrame, anio: int) -> pd.DataFrame:
    empresas = sorted(panel.empresa.unique())
    acc = accruals_flujo(panel).unstack("empresa")
    delta_acc = pd.Series(index=empresas, dtype=float)
    if anio in acc.index and anio - 1 in acc.index:
        delta_acc = (acc.loc[anio] - acc.loc[anio - 1]).reindex(empresas) * 100

    ben = beneish_panel(panel)
    if not ben.empty and anio in ben.anio.values:
        m = ben[ben.anio == anio].set_index("empresa")["M-Score"].reindex(empresas)
    else:
        m = pd.Series(index=empresas, dtype=float)

    p_anio = panel[panel.anio == anio].set_index("empresa").reindex(empresas)
    z2 = altman_z2(p_anio).reindex(empresas)

    out = pd.DataFrame({"Δ accruals (pp)": delta_acc, "M-Score": m, "Z''": z2}, index=empresas)
    out["Accruals"] = out["Δ accruals (pp)"].apply(estado_senal, args=(2.0, 1.5))
    out["Beneish"] = out["M-Score"].apply(estado_senal, args=(-1.78, -2.22))
    out["Altman Z''"] = out["Z''"].apply(estado_senal, args=(1.10, 2.60, False))
    return out


# ---------------------------------------------------------------- helpers de visualización
def fmt_ratio(nombre, v):
    if pd.isna(v):
        return "N/D"
    if nombre in {"Margen neto", "ROE", "ROA", "Margen operacional", "Margen bruto"}:
        return f"{v:.1%}"
    if nombre == "DSO (días)":
        return f"{v:.1f} días"
    if nombre in {"Rotación de activos", "Cobertura de intereses", "Deuda/Patrimonio", "Razón corriente"}:
        return f"{v:.2f}x"
    return f"{v:.2f}"


def color_estado(e):
    return {"OK": GREEN, "Vigilar": AMBER, "Alerta": RED, "N/D": GREY}.get(e, GREY)


def common_size(panel, anio, empresas_sel):
    d = panel[(panel.anio == anio) & (panel.empresa.isin(empresas_sel))].copy()
    er_cols = ["gross_profit", "sga", "depreciation", "ebit", "interest_expense", "net_income"]
    bs_cols = ["current_assets", "cash", "receivables", "inventory", "ppe",
               "total_liabilities", "current_liabilities", "long_term_debt", "equity"]
    er = d[["empresa"]].copy()
    for c in er_cols:
        er[c] = div(d[c], d.revenue)
    bs = d[["empresa"]].copy()
    for c in bs_cols:
        bs[c] = div(d[c], d.total_assets)
    return er.set_index("empresa"), bs.set_index("empresa")


# ---------------------------------------------------------------- barra lateral
st.sidebar.header("Parámetros")
subido = st.sidebar.file_uploader("Panel de estados financieros (CSV)", type="csv")
panel = cargar(subido)

if panel.empty:
    st.title("Taller 2 — Screener financiero de aerolíneas")
    st.info("Carga `panel_para_dashboard.csv` en la barra lateral para comenzar.")
    st.stop()

columnas_min = {
    "empresa", "anio", "revenue", "cogs", "gross_profit", "sga", "depreciation", "ebit",
    "interest_expense", "net_income", "total_assets", "current_assets", "cash", "receivables",
    "inventory", "ppe", "total_liabilities", "current_liabilities", "long_term_debt", "equity",
    "retained_earnings", "cfo"
}
faltan = sorted(columnas_min - set(panel.columns))
if faltan:
    st.error("Faltan columnas requeridas en el CSV: " + ", ".join(faltan))
    st.stop()

R = ratios(panel)
empresas = sorted(R.empresa.dropna().unique())
anios = sorted(int(x) for x in R.anio.dropna().unique())

foco_default = empresas.index("DAL") if "DAL" in empresas else min(2, len(empresas) - 1)
foco = st.sidebar.selectbox("Empresa foco", empresas, index=foco_default)
anio = st.sidebar.select_slider("Año", options=anios, value=anios[-1])
pares_disp = [e for e in empresas if e != foco]
pares = st.sidebar.multiselect("Pares de comparación", pares_disp, default=pares_disp)
if not pares:
    st.sidebar.warning("Selecciona al menos un par de comparación.")
    st.stop()

fila_df = R[(R.empresa == foco) & (R.anio == anio)]
if fila_df.empty:
    st.error(f"No hay datos para {foco} en {anio}.")
    st.stop()
fila = fila_df.iloc[0]
previo = R[(R.empresa == foco) & (R.anio == anio - 1)]
med_pares = R[(R.empresa.isin(pares)) & (R.anio == anio)].median(numeric_only=True)
X, pct, z, ranking = peer_screening(R[R.empresa.isin([foco] + pares)], anio)
alertas = tablero_alertas(panel[panel.empresa.isin([foco] + pares)], anio)

# ---------------------------------------------------------------- encabezado
st.title("Screener financiero de aerolíneas — Taller 2")
st.caption(
    f"{anio} | Empresa foco: {foco} | Pares: {', '.join(pares)}. "
    "Ranking principal por percentiles ponderados; z-score usado como desempate."
)

# ---------------------------------------------------------------- tabs
tab_resumen, tab_dupont, tab_tend, tab_alertas = st.tabs([
    "Resumen & ranking", "DuPont", "Tendencias & common-size", "Alertas"
])

# ================================================================ RESUMEN
with tab_resumen:
    pos = int(ranking.loc[foco, "Ranking"]) if foco in ranking.index else None
    score = ranking.loc[foco, "Puntaje (percentiles)"] if foco in ranking.index else np.nan
    cols_kpi = st.columns(5)
    cols_kpi[0].metric("Ranking", f"#{pos}" if pos else "N/D")
    cols_kpi[1].metric("Puntaje", f"{score:.2f}" if pd.notna(score) else "N/D")
    cols_kpi[2].metric("ROE", fmt_ratio("ROE", fila["ROE"]))
    cols_kpi[3].metric("Margen neto", fmt_ratio("Margen neto", fila["Margen neto"]))
    cols_kpi[4].metric("DSO", fmt_ratio("DSO (días)", fila["DSO (días)"]))

    st.subheader("Ranking final del peer screening")
    rshow = ranking.copy()
    rshow["Puntaje (percentiles)"] = rshow["Puntaje (percentiles)"].map(lambda x: f"{x:.2f}")
    rshow["Puntaje (z-scores)"] = rshow["Puntaje (z-scores)"].map(lambda x: f"{x:+.2f}")
    st.dataframe(rshow, width="stretch")

    if not ranking["Filtro profesor"].any():
        st.info(
            "El filtro absoluto del ejemplo (cobertura > 3x y razón corriente > 0,8x) no discrimina este grupo: "
            "ninguna empresa lo cumple. Por eso el ranking se basa en comparación relativa entre pares."
        )

    st.subheader("Peer screening: azul = mejor posición relativa; naranja = peor")
    if not pct.empty:
        Xc = X.loc[pct.index, COLS]
        texto = []
        for e in pct.index:
            fila_txt = []
            for c in COLS:
                fila_txt.append(fmt_ratio(c, Xc.loc[e, c]))
            texto.append(fila_txt)
        heat = go.Figure(go.Heatmap(
            z=pct[COLS].values,
            x=COLS,
            y=pct.index,
            text=texto,
            texttemplate="%{text}",
            colorscale=[[0, ACCENT], [0.5, "#F3E3D6"], [0.75, "#DCE5F2"], [1, NAVY]],
            zmin=0, zmax=1,
            showscale=False,
            hovertemplate="%{y}<br>%{x}<br>Percentil: %{z:.0%}<extra></extra>",
        ))
        heat.update_layout(height=310, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(heat, width="stretch")

    with st.expander("Criterio y pesos del ranking"):
        st.write(
            "Se mantienen los ocho ratios del laboratorio. En Margen neto, ROE, ROA, Rotación, Razón corriente y "
            "Cobertura de intereses, mayor es mejor. En Deuda/Patrimonio y DSO, menor es mejor."
        )
        pesos_df = pd.DataFrame({"Peso": pd.Series(PESOS), "Orientación": [
            "Menor es mejor" if c in MENOS_ES_MEJOR else "Mayor es mejor" for c in PESOS
        ]})
        st.dataframe(pesos_df.style.format({"Peso": "{:.0%}"}), width="stretch")

    # Interpretación automática de fortalezas y debilidades del foco.
    if foco in pct.index:
        p_foco = pct.loc[foco].sort_values(ascending=False)
        fortalezas = ", ".join(p_foco.head(3).index)
        debilidades = ", ".join(p_foco.tail(2).index)
        st.success(f"Fortalezas relativas de {foco}: {fortalezas}.")
        st.warning(f"Debilidades relativas de {foco}: {debilidades}.")

    ganador = ranking.index[0] if len(ranking) else None
    if ganador:
        st.markdown(
            f"**Recomendación breve:** el screener ubica a **{ganador}** en el primer lugar para {anio}. "
            "La decisión no debe basarse solo en el puntaje: se complementa con DuPont, tendencias y señales de alerta."
        )

# ================================================================ DUPONT
with tab_dupont:
    st.subheader(f"DuPont de 3 factores — {foco} vs. mediana de pares")
    izq, der = st.columns(2)
    comp = ["Margen neto", "Rotación de activos", "Multiplicador de capital", "ROE"]
    pct_cols = {"Margen neto", "ROE"}
    rel = pd.DataFrame({
        "Componente": comp,
        "Relativo": [div(fila[c], med_pares[c]) for c in comp],
        "Etiqueta": [
            (f"{fila[c]:.1%} vs {med_pares[c]:.1%}" if c in pct_cols else f"{fila[c]:.2f} vs {med_pares[c]:.2f}")
            for c in comp
        ],
    })
    fig = px.bar(rel, x="Componente", y="Relativo", text="Etiqueta", height=350,
                 color=rel.Componente.eq("ROE").map({True: "ROE", False: "Componente"}),
                 color_discrete_map={"ROE": NAVY, "Componente": ACCENT})
    fig.add_hline(y=1, line_dash="dash", line_color=GREY)
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Veces la mediana",
                      title=f"{foco}: qué componentes explican su ROE frente a los pares")
    izq.plotly_chart(fig, width="stretch")

    hist = R[R.empresa.isin([foco] + pares)].copy()
    hist["Serie"] = hist.empresa.where(hist.empresa == foco, "Pares")
    roe_t = hist.groupby(["anio", "Serie"], as_index=False)["ROE"].median()
    fig2 = px.line(roe_t, x="anio", y="ROE", color="Serie", markers=True, height=350,
                   color_discrete_map={foco: ACCENT, "Pares": GREY},
                   title=f"ROE: {foco} versus mediana de pares")
    fig2.update_yaxes(tickformat=".0%")
    fig2.update_xaxes(dtick=1, title=None)
    der.plotly_chart(fig2, width="stretch")

    st.subheader("DuPont de cinco factores")
    izq5, der5 = st.columns([5, 6])
    t5 = pd.DataFrame({"Factor": F5, foco: [fila[c] for c in F5], "Mediana pares": [med_pares[c] for c in F5]})
    t5["Relativo"] = t5[foco] / t5["Mediana pares"]
    izq5.dataframe(t5.set_index("Factor").style.format({foco: "{:.3f}", "Mediana pares": "{:.3f}", "Relativo": "{:.2f}x"}), width="stretch")

    if not fila["Interpretable"]:
        izq5.warning("EBIT o UAI negativos: las cargas tributaria y de intereses no son interpretables.")

    if previo.empty:
        der5.info("Se necesita el año anterior para atribuir la variación del ROE.")
    elif (fila[F5] <= 0).any() or (previo.iloc[0][F5] <= 0).any():
        der5.warning("La atribución logarítmica exige factores positivos en ambos años.")
    else:
        ant = previo.iloc[0]
        aporte = atribucion(ant, fila)
        wf = go.Figure(go.Waterfall(
            x=[f"ROE<br>{anio - 1}", "Carga<br>tribut.", "Carga<br>intereses", "Margen<br>EBIT", "Rotación", "Multipl.", f"ROE<br>{anio}"],
            measure=["absolute"] + ["relative"] * len(F5) + ["total"],
            y=[ant["ROE"]] + list(aporte.values) + [fila["ROE"]],
            text=[f"{ant['ROE']:.1%}"] + [f"{v * 100:+.1f} pp" for v in aporte.values] + [f"{fila['ROE']:.1%}"],
            textposition="outside",
            connector={"line": {"color": GREY, "dash": "dot"}},
            increasing={"marker": {"color": GREEN}},
            decreasing={"marker": {"color": RED}},
            totals={"marker": {"color": ACCENT}},
        ))
        mayor = aporte.abs().idxmax()
        wf.update_layout(title=f"El mayor aporte al cambio del ROE proviene de {mayor}", height=350,
                         showlegend=False, yaxis_tickformat=".0%")
        der5.plotly_chart(wf, width="stretch")

    st.subheader("Ratios frente a los pares")
    orden = ["ROE", "Margen neto", "ROA", "Margen operacional", "Rotación de activos", "Razón corriente",
             "Cobertura de intereses", "Deuda/Patrimonio", "DSO (días)", "Ventas/PP&E"]
    filas = []
    for c in orden:
        a, b = fila[c], med_pares[c]
        filas.append([c, fmt_ratio(c, a), fmt_ratio(c, b)])
    st.dataframe(pd.DataFrame(filas, columns=["Ratio", foco, "Mediana de pares"]).set_index("Ratio"), width="stretch")

# ================================================================ TENDENCIAS & COMMON SIZE
with tab_tend:
    st.subheader("Crecimiento de ventas — índice base 100")
    ventas = panel[panel.empresa.isin([foco] + pares)].pivot(index="anio", columns="empresa", values="revenue").sort_index()
    indice = 100 * ventas / ventas.iloc[0]
    idx_long = indice.reset_index().melt(id_vars="anio", var_name="empresa", value_name="Índice")
    cmap = {e: (ACCENT if e == foco else GREY) for e in [foco] + pares}
    fig_i = px.line(idx_long, x="anio", y="Índice", color="empresa", markers=True,
                    color_discrete_map=cmap, height=350,
                    title=f"{foco} frente a sus pares: trayectoria de ventas desde el año base")
    fig_i.add_hline(y=100, line_dash="dot", line_color=LIGHT)
    fig_i.update_xaxes(dtick=1, title=None)
    st.plotly_chart(fig_i, width="stretch")

    def cagr(s):
        s = s.dropna()
        if len(s) < 2 or s.iloc[0] <= 0 or s.iloc[-1] <= 0:
            return np.nan
        T = int(s.index[-1] - s.index[0])
        return (s.iloc[-1] / s.iloc[0]) ** (1 / T) - 1 if T > 0 else np.nan

    crec = ventas.apply(cagr).sort_values(ascending=False)
    cagr_df = crec.rename("CAGR").reset_index()
    fig_c = px.bar(cagr_df, x="empresa", y="CAGR", text=cagr_df.CAGR.map(lambda x: f"{x:.1%}"),
                   color=cagr_df.empresa.eq(foco).map({True: foco, False: "Pares"}),
                   color_discrete_map={foco: ACCENT, "Pares": NAVY},
                   title="CAGR de ventas: crecimiento anual compuesto")
    fig_c.update_yaxes(tickformat=".0%")
    fig_c.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(fig_c, width="stretch")

    st.subheader("Márgenes de la empresa foco versus mediana de pares")
    temp = R[R.empresa.isin([foco] + pares)].copy()
    for metrica in ["Margen operacional", "Margen neto"]:
        f = temp[temp.empresa == foco][["anio", metrica]].rename(columns={metrica: foco})
        m = temp[temp.empresa.isin(pares)].groupby("anio", as_index=False)[metrica].median().rename(columns={metrica: "Mediana pares"})
        mm = f.merge(m, on="anio").melt(id_vars="anio", var_name="Serie", value_name=metrica)
        fig_m = px.line(mm, x="anio", y=metrica, color="Serie", markers=True,
                        color_discrete_map={foco: ACCENT, "Mediana pares": GREY},
                        title=f"{metrica}: {foco} vs. mediana")
        fig_m.update_yaxes(tickformat=".0%")
        fig_m.update_xaxes(dtick=1, title=None)
        st.plotly_chart(fig_m, width="stretch")

    st.subheader(f"Common-size {anio}")
    er, bs = common_size(panel, anio, [foco] + pares)
    er_med = er.loc[pares].median()
    bs_med = bs.loc[pares].median()
    er_comp = pd.DataFrame({foco: er.loc[foco], "Mediana pares": er_med})
    bs_comp = pd.DataFrame({foco: bs.loc[foco], "Mediana pares": bs_med})
    c1, c2 = st.columns(2)
    c1.markdown("**Estado de resultados (% de ventas)**")
    c1.dataframe(er_comp.style.format("{:.1%}"), width="stretch")
    c2.markdown("**Balance (% de activos)**")
    c2.dataframe(bs_comp.style.format("{:.1%}"), width="stretch")

# ================================================================ ALERTAS
with tab_alertas:
    st.subheader(f"Señales de alerta — {anio}")
    af = alertas.loc[foco] if foco in alertas.index else pd.Series(dtype=float)
    s1 = af.get("Accruals", "N/D")
    s2 = af.get("Beneish", "N/D")
    s3 = af.get("Altman Z''", "N/D")
    ca, cb, cc = st.columns(3)
    ca.metric("Δ accruals", "N/D" if pd.isna(af.get("Δ accruals (pp)")) else f"{af['Δ accruals (pp)']:+.1f} pp")
    ca.markdown(f"**Estado:** :{ 'green' if s1 == 'OK' else 'orange' if s1 == 'Vigilar' else 'red' if s1 == 'Alerta' else 'gray' }[{s1}]")
    cb.metric("Beneish M-Score", "N/D" if pd.isna(af.get("M-Score")) else f"{af['M-Score']:.2f}")
    cb.markdown(f"**Estado:** :{ 'green' if s2 == 'OK' else 'orange' if s2 == 'Vigilar' else 'red' if s2 == 'Alerta' else 'gray' }[{s2}]")
    cc.metric("Altman Z''", "N/D" if pd.isna(af.get("Z''")) else "{:.2f}".format(af["Z''"]))
    cc.markdown(f"**Estado:** :{ 'green' if s3 == 'OK' else 'orange' if s3 == 'Vigilar' else 'red' if s3 == 'Alerta' else 'gray' }[{s3}]")

    st.caption("Beneish se muestra como N/D si falta alguno de los ocho índices necesarios; no se calcula un score parcial.")

    st.subheader("Tablero de alertas del grupo")
    orden_est = ["Accruals", "Beneish", "Altman Z''"]
    zmap = {"Alerta": 0, "Vigilar": 0.5, "OK": 1, "N/D": 0.25}
    zvals = alertas[orden_est].map(zmap.get)
    textos = []
    for e in alertas.index:
        textos.append([
            f"{alertas.loc[e, 'Accruals']}<br>{alertas.loc[e, 'Δ accruals (pp)']:+.1f} pp" if pd.notna(alertas.loc[e, "Δ accruals (pp)"]) else "N/D",
            f"{alertas.loc[e, 'Beneish']}<br>M={alertas.loc[e, 'M-Score']:.2f}" if pd.notna(alertas.loc[e, "M-Score"]) else "N/D",
            "{}<br>Z''={:.2f}".format(alertas.loc[e, "Altman Z''"], alertas.loc[e, "Z''"]) if pd.notna(alertas.loc[e, "Z''"]) else "N/D",
        ])
    fig_a = go.Figure(go.Heatmap(
        z=zvals.values,
        x=orden_est,
        y=alertas.index,
        text=textos,
        texttemplate="%{text}",
        colorscale=[[0, RED], [0.49, RED], [0.5, AMBER], [0.74, AMBER], [0.75, GREEN], [1, GREEN]],
        zmin=0, zmax=1, showscale=False,
        hoverinfo="skip",
    ))
    fig_a.update_layout(height=310, title="Rojo = alerta · ámbar = vigilar · verde = OK", margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_a, width="stretch")

    with st.expander("Cómo interpretar las señales"):
        st.markdown(
            "- **Accruals:** alerta si el ratio aumenta más de +2 pp frente al año anterior; vigilar sobre +1,5 pp.\n"
            "- **Beneish M-Score:** alerta sobre −1,78; vigilar sobre −2,22. Es una señal de revisión, no prueba de manipulación.\n"
            "- **Altman Z'':** alerta bajo 1,10; zona gris entre 1,10 y 2,60; sobre 2,60 se considera zona segura."
        )

st.caption("Fuente: panel de estados financieros cargado por el usuario. Dashboard desarrollado para Taller 2 — Finanzas Computacionales, UAI.")
