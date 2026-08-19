import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import statsmodels.api as sm

from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset
from statsmodels.stats.stattools import jarque_bera, durbin_watson


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Econometría Interactiva · UVigo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}
[data-testid="stSidebar"] {
    border-right: 1px solid #E5EAF0;
}
.hero {
    padding: 1.4rem 1.6rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #12355B 0%, #1F6E8C 100%);
    color: white;
    margin-bottom: 1rem;
}
.hero h1 {
    color: white;
    margin: 0;
    font-size: 2.2rem;
}
.hero p {
    margin: .5rem 0 0 0;
    opacity: .94;
}
.card {
    border: 1px solid #E1E7ED;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    background: white;
    min-height: 140px;
}
.kicker {
    font-size: .82rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    opacity: .85;
}
.small {
    font-size: .9rem;
    color: #5E6975;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATOS DEMO
# ============================================================

@st.cache_data
def make_demo_data():
    rng = np.random.default_rng(2026)
    n = 140

    publicidad = rng.gamma(3.2, 2.2, n) + 0.8
    superficie = np.clip(rng.normal(220, 75, n), 60, 480)
    precio = np.clip(rng.normal(18.5, 2.8, n), 11, 27)
    competencia = rng.integers(0, 9, n)
    empleados = np.clip(
        np.round(superficie / 42 + rng.normal(0, 1.4, n)),
        2,
        16
    ).astype(int)
    zona_centro = rng.binomial(1, 0.42, n)

    sigma = 8 + 0.035 * superficie
    error = rng.normal(0, sigma)

    ventas = (
        28
        + 4.6 * publicidad
        + 0.18 * superficie
        - 3.15 * precio
        - 2.4 * competencia
        + 1.7 * empleados
        + 9.0 * zona_centro
        + error
    )

    beneficio = ventas * rng.uniform(0.13, 0.24, n) - 0.45 * publicidad

    return pd.DataFrame({
        "ventas_miles_eur": np.round(ventas, 2),
        "publicidad_miles_eur": np.round(publicidad, 2),
        "superficie_m2": np.round(superficie, 1),
        "precio_medio_eur": np.round(precio, 2),
        "competidores_1km": competencia,
        "empleados": empleados,
        "zona_centro": zona_centro,
        "beneficio_miles_eur": np.round(beneficio, 2),
    })


def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def fit_ols(df, y, xs, robust=False):
    dat = (
        df[[y] + list(xs)]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )
    X = sm.add_constant(dat[list(xs)], has_constant="add")
    model = sm.OLS(dat[y], X).fit()

    if robust:
        model = model.get_robustcov_results(cov_type="HC1")

    return model, dat


def coef_table(model):
    names = list(model.model.exog_names)
    conf = np.asarray(model.conf_int())

    return pd.DataFrame({
        "Variable": names,
        "Coeficiente": np.asarray(model.params),
        "Error típico": np.asarray(model.bse),
        "t": np.asarray(model.tvalues),
        "p-valor": np.asarray(model.pvalues),
        "IC 95% inf.": conf[:, 0],
        "IC 95% sup.": conf[:, 1],
    })


def equation_text(model, y):
    names = list(model.model.exog_names)
    params = np.asarray(model.params)

    pieces = []
    for name, b in zip(names, params):
        if name == "const":
            pieces.append(f"{b:.3f}")
        else:
            sign = "+" if b >= 0 else "-"
            pieces.append(f" {sign} {abs(b):.3f}·{name}")

    return f"{y} = " + "".join(pieces)


def gretl_script(y, xs):
    regressors = " ".join(xs)
    return f"""# ECONOMETRÍA · SCRIPT BASE GRETl

open datos.csv

# 1. Descriptivos
summary {y} {regressors}
corr {y} {regressors}

# 2. Estimación por MCO
ols {y} const {regressors}

# 3. Diagnóstico
vif
modtest --normality
modtest --squares
modtest --white

# 4. Guarda el script para reproducibilidad
"""


# ============================================================
# NAVEGACIÓN
# ============================================================

st.sidebar.markdown("## 📈 Econometría")
st.sidebar.caption("ADE · Universidade de Vigo")

section = st.sidebar.radio(
    "Navegación",
    [
        "🏠 Inicio",
        "🧭 Temario",
        "📈 Laboratorio MCO",
        "🔎 Interpretar coeficientes",
        "🧪 Contrastes e IC",
        "🩺 Diagnóstico",
        "⌨️ Gretl",
        "✅ Mini-test",
        "🗂️ Microproyecto",
    ],
)

st.sidebar.divider()
st.sidebar.caption(
    "Material complementario de clase · "
    "Prof. Alessandro Indelicato"
)


# ============================================================
# INICIO
# ============================================================

if section == "🏠 Inicio":

    st.markdown("""
    <div class="hero">
      <div class="kicker">Econometría · ADE · 2026/27</div>
      <h1>Econometría Interactiva</h1>
      <p>De la pregunta económica al modelo, la estimación,
      el diagnóstico y la interpretación.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bloques", "6")
    c2.metric("Laboratorios", "5")
    c3.metric("Mini-test", "10 preguntas")
    c4.metric("Caso demo", "ADE")

    st.subheader("Cómo vamos a trabajar")

    cols = st.columns(4)

    cards = [
        (
            "1 · Pregunta",
            "Formular una cuestión económica medible y decidir qué variable queremos explicar."
        ),
        (
            "2 · Modelo",
            "Traducir la pregunta a una ecuación, elegir variables y explicitar supuestos."
        ),
        (
            "3 · Evidencia",
            "Estimar por MCO, contrastar hipótesis y revisar la calidad del modelo."
        ),
        (
            "4 · Decisión",
            "Interpretar magnitudes, incertidumbre y límites sin confundir asociación con causalidad."
        ),
    ]

    for col, (title, body) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="card"><b>{title}</b><p>{body}</p></div>',
                unsafe_allow_html=True
            )

    st.subheader("Ruta del curso")

    st.markdown("""
    **Datos y Gretl** → **Regresión simple y múltiple** →
    **MCO y ANOVA** → **Inferencia** → **Diagnóstico** →
    **Informe econométrico reproducible**
    """)

    st.info(
        "La app no sustituye Gretl. Sirve para experimentar con los conceptos, "
        "comprobar interpretaciones y entender qué hay detrás de la salida del programa."
    )


# ============================================================
# TEMARIO
# ============================================================

elif section == "🧭 Temario":

    st.title("🧭 Temario y sesiones")
    st.write(
        "La secuencia sigue el flujo completo de un análisis econométrico aplicado."
    )

    sessions = [
        {
            "title": "Sesión 1 · Gretl, datos y descriptivos",
            "goal": "Preparar una base reproducible y mirar los datos antes de estimar.",
            "topics": [
                "Estructuras de datos",
                "Muestra y unidad de observación",
                "Descriptivos",
                "Correlación",
                "Gráficos",
                "Transformaciones y ficticias",
            ],
            "formula": r"\bar{x}=\frac{1}{n}\sum_{i=1}^n x_i,\qquad s^2=\frac{1}{n-1}\sum_{i=1}^n(x_i-\bar{x})^2",
            "gretl": "open datos.csv\nsummary\ncorr\nscatters y x1 x2",
            "product": "Dataset limpio + descriptivos + gráfico comentado",
        },
        {
            "title": "Sesión 2 · MRLC y estimación por MCO",
            "goal": "Entender qué estima una regresión y qué significa ceteris paribus.",
            "topics": [
                "Regresión simple y múltiple",
                "Perturbación",
                "Exogeneidad",
                "MCO",
                "Valores ajustados",
                "Residuos",
                "Interpretación económica",
            ],
            "formula": r"y_i=\beta_0+\beta_1x_{1i}+\cdots+\beta_kx_{ki}+u_i",
            "gretl": "ols y const x1 x2 x3",
            "product": "Modelo formulado, estimado e interpretado",
        },
        {
            "title": "Sesión 3 · MCO matricial, ANOVA y forma funcional",
            "goal": "Conectar salida econométrica, geometría, varianza y unidades.",
            "topics": [
                "Notación matricial",
                "Proyección",
                "ANOVA",
                "R²",
                "R² ajustado",
                "Escala",
                "Logaritmos",
                "Elasticidades",
            ],
            "formula": r"\widehat{\beta}=(X'X)^{-1}X'y,\qquad SCT=SCR+SCE",
            "gretl": "genr ly = log(y)\ngenr lx1 = log(x1)\nols ly const lx1 x2",
            "product": "Comparación razonada de especificaciones",
        },
        {
            "title": "Sesión 4 · Inferencia, intervalos y restricciones",
            "goal": "Incorporar incertidumbre a las afirmaciones econométricas.",
            "topics": [
                "Error típico",
                "t de Student",
                "p-valor",
                "Intervalos de confianza",
                "Contrastes F",
                "Restricciones lineales",
            ],
            "formula": r"t=\frac{\widehat{\beta}_j-\beta_{j,0}}{se(\widehat{\beta}_j)}",
            "gretl": "ols y const x1 x2 x3\nrestrict\n b[x1] = b[x2]\nend restrict",
            "product": "Decisión estadística + interpretación económica",
        },
        {
            "title": "Sesión 5 · Multicolinealidad y diagnóstico",
            "goal": "Detectar problemas de inferencia o especificación.",
            "topics": [
                "VIF",
                "Normalidad",
                "RESET",
                "Heterocedasticidad",
                "White",
                "Breusch–Pagan",
                "Autocorrelación",
                "Errores robustos",
            ],
            "formula": r"VIF_j=\frac{1}{1-R_j^2},\qquad Var(u_i\mid X)=\sigma_i^2",
            "gretl": "vif\nmodtest --normality\nmodtest --squares\nmodtest --white",
            "product": "Tabla de diagnóstico y decisiones justificadas",
        },
        {
            "title": "Sesión 6 · Taller final e informe econométrico",
            "goal": "Cerrar el ciclo completo del análisis.",
            "topics": [
                "Reproducibilidad",
                "Selección del modelo",
                "Robustez",
                "Interpretación",
                "Limitaciones",
                "Asociación vs causalidad",
                "Defensa oral",
            ],
            "formula": r"\text{Pregunta}\rightarrow\text{Datos}\rightarrow\text{Modelo}\rightarrow\text{Diagnóstico}\rightarrow\text{Conclusión}",
            "gretl": "# El script final debe ejecutarse de principio a fin sin errores",
            "product": "Informe + base + script Gretl + tabla principal + gráfico",
        },
    ]

    for s in sessions:
        with st.expander(
            s["title"],
            expanded=s["title"].startswith("Sesión 1")
        ):
            st.markdown(f"**Objetivo:** {s['goal']}")
            st.markdown("**Conceptos:** " + " · ".join(s["topics"]))
            st.latex(s["formula"])
            st.markdown("**Gretl**")
            st.code(s["gretl"], language="text")
            st.success("Producto de aula: " + s["product"])

    st.warning(
        "Regla transversal: una salida estadística nunca termina en el p-valor; "
        "termina en una interpretación con magnitud, unidades, incertidumbre y límites."
    )


# ============================================================
# LABORATORIO MCO
# ============================================================

elif section == "📈 Laboratorio MCO":

    st.title("📈 Laboratorio de regresión MCO")
    st.write(
        "Carga tus datos o utiliza el caso demo de establecimientos comerciales."
    )

    uploaded = st.file_uploader("CSV del alumno", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            source = "CSV cargado"
        except Exception as exc:
            st.error(f"No pude leer el CSV: {exc}")
            st.stop()
    else:
        df = make_demo_data()
        source = "Caso demo ADE"

    st.caption(
        f"Fuente activa: **{source}** · "
        f"{len(df)} observaciones · {df.shape[1]} variables"
    )

    st.dataframe(df.head(12), use_container_width=True)

    num = numeric_columns(df)

    if len(num) < 2:
        st.error("La base necesita al menos dos variables numéricas.")
        st.stop()

    default_y = (
        num.index("ventas_miles_eur")
        if "ventas_miles_eur" in num
        else 0
    )

    y = st.selectbox(
        "Variable dependiente (Y)",
        num,
        index=default_y
    )

    x_candidates = [c for c in num if c != y]

    defaults = [
        c
        for c in [
            "publicidad_miles_eur",
            "superficie_m2",
            "precio_medio_eur",
            "competidores_1km",
        ]
        if c in x_candidates
    ]

    xs = st.multiselect(
        "Variables explicativas (X)",
        x_candidates,
        default=defaults[:4] or x_candidates[:2],
    )

    robust = st.toggle(
        "Usar errores estándar robustos HC1",
        value=False
    )

    if not xs:
        st.info("Selecciona al menos una variable explicativa.")
        st.stop()

    try:
        model, dat = fit_ols(df, y, xs, robust=robust)
    except Exception as exc:
        st.error(f"No se pudo estimar el modelo: {exc}")
        st.stop()

    st.subheader("1. Modelo estimado")
    st.code(equation_text(model, y), language="text")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("N", f"{int(model.nobs)}")
    m2.metric("R²", f"{model.rsquared:.3f}")
    m3.metric("R² ajustado", f"{model.rsquared_adj:.3f}")

    fp = model.f_pvalue
    m4.metric(
        "F · p-valor",
        f"{fp:.3g}" if np.isfinite(fp) else "—"
    )

    tab = coef_table(model)

    st.dataframe(
        tab.style.format({
            "Coeficiente": "{:.4f}",
            "Error típico": "{:.4f}",
            "t": "{:.3f}",
            "p-valor": "{:.4f}",
            "IC 95% inf.": "{:.4f}",
            "IC 95% sup.": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("2. Interpretación guiada")

    coef_names = [
        x
        for x in xs
        if x in tab["Variable"].tolist()
    ]

    selected = st.selectbox(
        "Coeficiente que quieres interpretar",
        coef_names
    )

    row = tab.loc[tab["Variable"] == selected].iloc[0]

    b = row["Coeficiente"]
    p = row["p-valor"]
    direction = "aumento" if b >= 0 else "disminución"

    st.info(
        f"Manteniendo constantes las demás variables, "
        f"un incremento de 1 unidad en **{selected}** se asocia con un "
        f"**{direction} de {abs(b):.3f} unidades de {y}**. "
        f"El p-valor es {p:.4f}. "
        "Esto describe una asociación condicional; por sí solo no demuestra causalidad."
    )

    st.subheader("3. Ajuste y residuos")

    plot_df = pd.DataFrame({
        "Ajustado": np.asarray(model.fittedvalues),
        "Residuo": np.asarray(model.resid),
        "Observado": dat[y].to_numpy(),
    })

    fig1 = px.scatter(
        plot_df,
        x="Ajustado",
        y="Residuo",
        hover_data=["Observado"],
        title="Residuos frente a valores ajustados",
    )
    fig1.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(
        plot_df,
        x="Observado",
        y="Ajustado",
        title="Valores observados frente a ajustados",
    )

    lo = float(
        min(
            plot_df["Observado"].min(),
            plot_df["Ajustado"].min()
        )
    )
    hi = float(
        max(
            plot_df["Observado"].max(),
            plot_df["Ajustado"].max()
        )
    )

    fig2.add_shape(
        type="line",
        x0=lo,
        y0=lo,
        x1=hi,
        y1=hi,
        line=dict(dash="dash"),
    )

    st.plotly_chart(fig2, use_container_width=True)

    out = dat.copy()
    out["y_ajustada"] = np.asarray(model.fittedvalues)
    out["residuo"] = np.asarray(model.resid)

    st.download_button(
        "Descargar datos + ajustes + residuos",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="resultado_mco.csv",
        mime="text/csv",
    )

    with st.expander("Resumen técnico completo"):
        st.text(model.summary().as_text())


# ============================================================
# INTERPRETACIÓN
# ============================================================

elif section == "🔎 Interpretar coeficientes":

    st.title("🔎 Interpretar coeficientes")
    st.write(
        "Entrena la traducción de una fórmula a una frase económica correcta."
    )

    form = st.selectbox(
        "Forma funcional",
        [
            "Nivel–nivel: Y = β0 + β1 X",
            "Log–nivel: ln(Y) = β0 + β1 X",
            "Nivel–log: Y = β0 + β1 ln(X)",
            "Log–log: ln(Y) = β0 + β1 ln(X)",
        ],
    )

    b = st.number_input(
        "Coeficiente estimado β̂₁",
        value=0.2500,
        step=0.01,
        format="%.4f",
    )

    delta = st.number_input(
        "Cambio en X que quieres interpretar",
        value=1.0,
        step=0.5,
    )

    st.subheader("Interpretación")

    if form.startswith("Nivel–nivel"):
        change = b * delta
        st.latex(r"\Delta Y \approx \widehat{\beta}_1\Delta X")
        st.success(
            f"Un aumento de {delta:g} unidades en X se asocia con "
            f"un cambio de {change:.4g} unidades en Y, ceteris paribus."
        )

    elif form.startswith("Log–nivel"):
        approx_pct = 100 * b * delta
        exact_pct = 100 * (math.exp(b * delta) - 1)

        st.latex(
            r"\%\Delta Y \approx 100\widehat{\beta}_1\Delta X"
        )

        st.success(
            f"Aproximación: X +{delta:g} → Y cambia aproximadamente "
            f"{approx_pct:.2f}%. Cambio exacto: {exact_pct:.2f}%."
        )

    elif form.startswith("Nivel–log"):
        pct_x = st.number_input(
            "Cambio porcentual de X",
            value=10.0,
            step=1.0,
        )

        change_y = b * (pct_x / 100)

        st.latex(
            r"\Delta Y \approx \widehat{\beta}_1\frac{\%\Delta X}{100}"
        )

        st.success(
            f"Un aumento del {pct_x:g}% en X se asocia con un "
            f"cambio de {change_y:.4g} unidades en Y."
        )

    else:
        pct_x = st.number_input(
            "Cambio porcentual de X",
            value=10.0,
            step=1.0,
        )

        approx_pct_y = b * pct_x

        st.latex(
            r"\%\Delta Y \approx \widehat{\beta}_1\,\%\Delta X"
        )

        st.success(
            f"β̂₁ es una elasticidad: un aumento del {pct_x:g}% en X "
            f"se asocia con un cambio aproximado del {approx_pct_y:.2f}% en Y."
        )

    st.warning(
        "Tres elementos deben aparecer siempre: magnitud, unidad y ceteris paribus. "
        "La significación estadística se comenta aparte."
    )


# ============================================================
# CONTRASTES
# ============================================================

elif section == "🧪 Contrastes e IC":

    st.title("🧪 Contrastes de hipótesis e intervalos")
    st.write(
        "Experimenta con H₀ y observa cómo cambian t, p-valor y decisión."
    )

    df = make_demo_data()
    num = numeric_columns(df)

    default_y = (
        num.index("ventas_miles_eur")
        if "ventas_miles_eur" in num
        else 0
    )

    y = st.selectbox(
        "Y",
        num,
        index=default_y,
        key="contrast_y",
    )

    xs_all = [c for c in num if c != y]

    defaults = [
        c
        for c in [
            "publicidad_miles_eur",
            "superficie_m2",
            "precio_medio_eur",
            "competidores_1km",
        ]
        if c in xs_all
    ]

    xs = st.multiselect(
        "X",
        xs_all,
        default=defaults[:4] or xs_all[:2],
        key="contrast_x",
    )

    if not xs:
        st.stop()

    model, dat = fit_ols(df, y, xs)

    coef = st.selectbox(
        "Parámetro a contrastar",
        xs
    )

    h0 = st.number_input(
        "Valor bajo H₀: βj =",
        value=0.0,
        step=0.1,
    )

    alpha = st.select_slider(
        "Nivel de significación α",
        options=[0.10, 0.05, 0.01],
        value=0.05,
    )

    idx = list(model.model.exog_names).index(coef)

    b = float(model.params[idx])
    se = float(model.bse[idx])
    df_resid = float(model.df_resid)

    t_stat = (b - h0) / se
    pval = 2 * stats.t.sf(abs(t_stat), df_resid)

    crit = stats.t.ppf(
        1 - alpha / 2,
        df_resid
    )

    ci = (
        b - crit * se,
        b + crit * se,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("β̂", f"{b:.4f}")
    c2.metric("SE", f"{se:.4f}")
    c3.metric("t", f"{t_stat:.3f}")
    c4.metric("p-valor", f"{pval:.4g}")

    st.latex(
        r"t=\frac{\widehat{\beta}_j-\beta_{j,0}}{se(\widehat{\beta}_j)}"
    )

    st.write(
        f"IC {(1-alpha)*100:.0f}%: "
        f"**[{ci[0]:.4f}, {ci[1]:.4f}]**"
    )

    if pval < alpha:
        st.success(
            f"Como p = {pval:.4g} < α = {alpha:.2f}, rechazamos H₀."
        )
    else:
        st.info(
            f"Como p = {pval:.4g} ≥ α = {alpha:.2f}, no rechazamos H₀."
        )

    st.caption(
        "«No rechazar H₀» no significa demostrar que H₀ sea verdadera."
    )


# ============================================================
# DIAGNÓSTICO
# ============================================================

elif section == "🩺 Diagnóstico":

    st.title("🩺 Diagnóstico del modelo")
    st.write(
        "Cada prueba responde a una pregunta distinta sobre el modelo."
    )

    df = make_demo_data()
    num = numeric_columns(df)

    default_y = (
        num.index("ventas_miles_eur")
        if "ventas_miles_eur" in num
        else 0
    )

    y = st.selectbox(
        "Y",
        num,
        index=default_y,
        key="diag_y",
    )

    xs_all = [c for c in num if c != y]

    defaults = [
        c
        for c in [
            "publicidad_miles_eur",
            "superficie_m2",
            "precio_medio_eur",
            "competidores_1km",
        ]
        if c in xs_all
    ]

    xs = st.multiselect(
        "X",
        xs_all,
        default=defaults[:4] or xs_all[:2],
        key="diag_x",
    )

    if not xs:
        st.stop()

    model, dat = fit_ols(df, y, xs)

    resid = np.asarray(model.resid)
    exog = np.asarray(model.model.exog)

    st.subheader("Multicolinealidad · VIF")

    vif_rows = []

    for i, name in enumerate(model.model.exog_names):

        if name == "const":
            continue

        try:
            vif = variance_inflation_factor(exog, i)
        except Exception:
            vif = np.nan

        vif_rows.append({
            "Variable": name,
            "VIF": vif
        })

    vif_df = pd.DataFrame(vif_rows)

    st.dataframe(
        vif_df.style.format({"VIF": "{:.2f}"}),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Los VIF elevados alertan de pérdida de precisión por colinealidad. "
        "No existe un umbral universal que sustituya el juicio sustantivo."
    )

    st.subheader("Pruebas de diagnóstico")

    rows = []

    try:
        lm, lm_p, f, f_p = het_breuschpagan(resid, exog)
        rows.append([
            "Breusch–Pagan",
            "Homoscedasticidad",
            lm,
            lm_p,
        ])
    except Exception:
        pass

    try:
        lm, lm_p, f, f_p = het_white(resid, exog)
        rows.append([
            "White",
            "Homoscedasticidad / especificación",
            lm,
            lm_p,
        ])
    except Exception:
        pass

    try:
        jb, jb_p, skew, kurt = jarque_bera(resid)
        rows.append([
            "Jarque–Bera",
            "Normalidad de residuos",
            jb,
            jb_p,
        ])
    except Exception:
        pass

    try:
        reset = linear_reset(
            model,
            power=2,
            use_f=True
        )

        rows.append([
            "RESET",
            "Forma funcional",
            float(reset.fvalue),
            float(reset.pvalue),
        ])
    except Exception:
        pass

    test_df = pd.DataFrame(
        rows,
        columns=[
            "Prueba",
            "Pregunta",
            "Estadístico",
            "p-valor",
        ],
    )

    st.dataframe(
        test_df.style.format({
            "Estadístico": "{:.3f}",
            "p-valor": "{:.4f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    dw = durbin_watson(resid)

    st.metric(
        "Durbin–Watson",
        f"{dw:.3f}"
    )

    st.caption(
        "DW ≈ 2 es compatible con ausencia de autocorrelación AR(1), "
        "pero su uso debe ser coherente con la estructura de los datos."
    )

    plot_df = pd.DataFrame({
        "Ajustado": np.asarray(model.fittedvalues),
        "Residuo": resid,
    })

    fig = px.scatter(
        plot_df,
        x="Ajustado",
        y="Residuo",
        title="Residuos frente a valores ajustados",
    )

    fig.add_hline(
        y=0,
        line_dash="dash"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "Si una prueba detecta un problema, la pregunta importante es: "
        "¿qué consecuencia tiene para la estimación o la inferencia "
        "y qué actuación está justificada?"
    )


# ============================================================
# GRETL
# ============================================================

elif section == "⌨️ Gretl":

    st.title("⌨️ Generador de script Gretl")
    st.write(
        "Selecciona un modelo y genera una plantilla reproducible."
    )

    df = make_demo_data()
    num = numeric_columns(df)

    default_y = (
        num.index("ventas_miles_eur")
        if "ventas_miles_eur" in num
        else 0
    )

    y = st.selectbox(
        "Variable dependiente",
        num,
        index=default_y,
        key="gretl_y",
    )

    xs_all = [c for c in num if c != y]

    defaults = [
        c
        for c in [
            "publicidad_miles_eur",
            "superficie_m2",
            "precio_medio_eur",
            "competidores_1km",
        ]
        if c in xs_all
    ]

    xs = st.multiselect(
        "Explicativas",
        xs_all,
        default=defaults[:4] or xs_all[:2],
        key="gretl_x",
    )

    if xs:
        script = gretl_script(y, xs)

        st.code(
            script,
            language="text"
        )

        st.download_button(
            "Descargar script .inp",
            data=script.encode("utf-8"),
            file_name="modelo_gretl.inp",
            mime="text/plain",
        )

    st.subheader("Comandos que debes reconocer")

    st.markdown("""
    - `open` — abrir una base.
    - `summary` — estadísticos descriptivos.
    - `corr` — correlaciones.
    - `genr` — generar transformaciones.
    - `ols` — estimar por MCO.
    - `restrict` — contrastar restricciones lineales.
    - `vif` — diagnóstico de multicolinealidad.
    - `modtest` — pruebas de especificación y diagnóstico.
    - `smpl` — modificar la muestra activa.
    """)

    st.warning(
        "La plantilla debe adaptarse a los nombres de variables "
        "y a la estructura real de cada base."
    )


# ============================================================
# QUIZ
# ============================================================

elif section == "✅ Mini-test":

    st.title("✅ Mini-test de Econometría")
    st.write(
        "10 preguntas rápidas. La corrección aparece al enviar."
    )

    questions = [
        (
            "¿Qué representa β₁ en una regresión múltiple?",
            [
                "La correlación simple entre Y y X₁",
                "El cambio medio en Y asociado a una unidad de X₁, manteniendo constantes las demás X",
                "La probabilidad de que X₁ cause Y",
                "El error de predicción",
            ],
            1,
        ),
        (
            "¿Cuál es el criterio de MCO?",
            [
                "Maximizar R² ajustado",
                "Minimizar la suma de residuos",
                "Minimizar la suma de residuos al cuadrado",
                "Minimizar el p-valor",
            ],
            2,
        ),
        (
            "Si p = 0,03 y α = 0,05 en H₀: β=0:",
            [
                "No rechazamos H₀",
                "Rechazamos H₀",
                "Demostramos causalidad",
                "El coeficiente es económicamente importante",
            ],
            1,
        ),
        (
            "Un R² elevado implica necesariamente que:",
            [
                "El modelo es causal",
                "Todos los coeficientes son significativos",
                "El modelo reproduce una fracción alta de la variación muestral de Y",
                "No hay heterocedasticidad",
            ],
            2,
        ),
        (
            "¿Qué diagnostica principalmente el VIF?",
            [
                "Heterocedasticidad",
                "Multicolinealidad",
                "Autocorrelación",
                "Normalidad",
            ],
            1,
        ),
        (
            "Con heterocedasticidad y exogeneidad, MCO:",
            [
                "Siempre es sesgado",
                "No puede estimarse",
                "Puede seguir siendo insesgado/consistente, pero falla la fórmula clásica de varianza",
                "Se vuelve automáticamente causal",
            ],
            2,
        ),
        (
            "En un modelo log-log, β₁ se interpreta aproximadamente como:",
            [
                "Una diferencia de medias",
                "Una elasticidad",
                "Una odds ratio",
                "Un error estándar",
            ],
            1,
        ),
        (
            "¿Qué significa ceteris paribus?",
            [
                "Todo cambia a la vez",
                "La muestra es aleatoria",
                "Mantener constantes los demás factores incluidos en el modelo",
                "Eliminar el término de error",
            ],
            2,
        ),
        (
            "No rechazar H₀ significa que:",
            [
                "H₀ ha sido demostrada",
                "Los datos no aportan evidencia suficiente para rechazarla al nivel elegido",
                "H₁ es falsa",
                "El efecto es exactamente cero",
            ],
            1,
        ),
        (
            "Antes de hablar de causalidad, además de una regresión, necesitamos:",
            [
                "Un R² mayor que 0,80",
                "Una estrategia de identificación y supuestos creíbles",
                "Más variables siempre",
                "Un p-valor menor que 0,10",
            ],
            1,
        ),
    ]

    with st.form("quiz_form"):
        responses = []

        for i, (q, opts, ans) in enumerate(
            questions,
            start=1
        ):
            st.markdown(f"**{i}. {q}**")

            responses.append(
                st.radio(
                    "Respuesta",
                    opts,
                    key=f"q{i}",
                    index=None,
                    label_visibility="collapsed",
                )
            )

        submitted = st.form_submit_button(
            "Corregir"
        )

    if submitted:

        score = 0

        for response, (_, opts, ans) in zip(
            responses,
            questions
        ):
            if response == opts[ans]:
                score += 1

        st.metric(
            "Resultado",
            f"{score}/10"
        )

        if score >= 8:
            st.success(
                "Muy bien: dominio sólido de los conceptos básicos."
            )
        elif score >= 5:
            st.info(
                "Base suficiente, pero conviene repasar las preguntas falladas."
            )
        else:
            st.warning(
                "Conviene volver al temario y repetir el test después."
            )

        with st.expander(
            "Ver respuestas correctas"
        ):
            for i, (q, opts, ans) in enumerate(
                questions,
                start=1
            ):
                st.write(
                    f"**{i}.** {opts[ans]}"
                )


# ============================================================
# MICROPROYECTO
# ============================================================

elif section == "🗂️ Microproyecto":

    st.title("🗂️ Guía del microproyecto")
    st.write(
        "La evaluación reproduce el flujo de un análisis econométrico real."
    )

    steps = [
        (
            "1. Pregunta de investigación",
            "Concreta, medible y susceptible de responderse con un modelo econométrico."
        ),
        (
            "2. Justificación económica",
            "Explica por qué importa y qué relación esperas encontrar."
        ),
        (
            "3. Unidad de observación y muestra",
            "Define unidad, N, periodo y ámbito geográfico."
        ),
        (
            "4. Fuente y variables",
            "Documenta procedencia, unidades, definiciones, signos esperados y transformaciones."
        ),
        (
            "5. Modelo inicial",
            "Escribe la especificación con nombres reales y justifica controles."
        ),
        (
            "6. Descriptivos y gráfico",
            "Media, desviación, mínimo, máximo, N y lectura sustantiva del gráfico."
        ),
        (
            "7. Estimación e inferencia",
            "Interpreta coeficientes, errores típicos, p-valores, R² e IC."
        ),
        (
            "8. Contraste y diagnóstico",
            "Formula H₀/H₁ y aplica al menos dos diagnósticos relevantes."
        ),
        (
            "9. Robustez, límites y conclusión",
            "Distingue asociación de causalidad y deja el análisis reproducible."
        ),
    ]

    for title, body in steps:
        with st.expander(title):
            st.write(body)

    st.subheader("Checklist de entrega")

    checks = [
        "Pregunta y justificación",
        "Definición de Y y X",
        "Base de datos",
        "Script Gretl reproducible",
        "Tabla de descriptivos",
        "Gráfico relevante",
        "Salida principal MCO",
        "Contraste de hipótesis",
        "Dos diagnósticos justificados",
        "Conclusión con limitaciones",
    ]

    for i, item in enumerate(checks):
        st.checkbox(
            item,
            key=f"check_{i}"
        )

    template = """# Microproyecto de Econometría

## 1. Pregunta de investigación

## 2. Justificación económica

## 3. Unidad de observación y muestra

## 4. Fuente y variables

## 5. Modelo econométrico inicial

## 6. Estadísticos descriptivos y gráfico

## 7. Resultados MCO e interpretación

## 8. Contraste de hipótesis

## 9. Diagnóstico del modelo

## 10. Robustez y especificación alternativa

## 11. Limitaciones

## 12. Conclusión

## Archivos reproducibles
- Base de datos:
- Script Gretl:
- Tabla principal:
- Gráfico:
"""

    st.download_button(
        "Descargar plantilla del informe (.md)",
        data=template.encode("utf-8"),
        file_name="plantilla_microproyecto.md",
        mime="text/markdown",
    )

    st.info(
        "En la defensa debes poder interpretar cualquier coeficiente, "
        "explicar un p-valor, justificar las pruebas de diagnóstico "
        "y modificar una especificación en Gretl."
    )
