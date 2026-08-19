import math
import csv
import io
import random
import re
import smtplib
import ssl
import uuid
import json
import tempfile
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
from email.message import EmailMessage
from pathlib import Path

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
# CARGA DE DATOS DEL ALUMNO
# ============================================================

def load_uploaded_dataset(uploaded_file):
    """Lee una base CSV o XLSX subida por el usuario."""
    if uploaded_file is None:
        return None

    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        # Intento estándar; si el separador es ;, pandas lo detecta con engine=python.
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=None, engine="python")

    if filename.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, engine="openpyxl")

    raise ValueError("Formato no compatible. Utiliza CSV o XLSX.")


def require_uploaded_dataset(key, label="Sube la base de datos"):
    """
    Obliga a subir una base antes de utilizar un laboratorio.
    No existe ningún dataset demo o automático.
    """
    uploaded = st.file_uploader(
        label,
        type=["csv", "xlsx"],
        key=key,
        help="La base se utiliza únicamente durante esta sesión de la aplicación.",
    )

    if uploaded is None:
        st.info(
            "⬆️ Sube una base de datos en formato CSV o XLSX para continuar."
        )
        st.stop()

    try:
        df = load_uploaded_dataset(uploaded)
    except Exception as exc:
        st.error("No se pudo leer la base de datos.")
        st.code(str(exc), language="text")
        st.stop()

    if df is None or df.empty:
        st.error("La base está vacía.")
        st.stop()

    return df, uploaded.name


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


def gretl_script(y, xs, filename="datos.csv"):
    regressors = " ".join(xs)
    return f"""# ECONOMETRÍA · SCRIPT BASE GRETl

open {filename}

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
# BANCO DE PREGUNTAS Y ENVÍO DE TEST
# ============================================================

BANK_FILENAME = "Banco_preguntas_MooVi_Econometria.xlsx"
TEACHER_EMAIL = "alessandro.indelicato.a@gmail.com"

FICHA_LABELS = {
    "Ficha00": "Ficha00 · Introducción a Gretl y entorno de trabajo",
    "Ficha01": "Ficha01 · Datos, gráficos, estadística y álgebra en Gretl",
    "Ficha02": "Ficha02 · Regresión lineal y estimación por MCO",
    "Ficha03": "Ficha03 · Propiedades MCO, especificación y cambios de escala",
    "Ficha04": "Ficha04 · Inferencia, contrastes e intervalos de confianza",
    "Ficha05": "Ficha05 · Multicolinealidad y especificación",
    "Ficha06": "Ficha06 · Diagnóstico general del modelo econométrico",
}

LABEL_TO_FICHA = {v: k for k, v in FICHA_LABELS.items()}


def _safe_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


@st.cache_data(show_spinner=False)
def load_question_bank():
    """Carga las 1.750 preguntas desde la hoja 'Todas' del banco Excel."""
    bank_path = Path(__file__).resolve().parent / BANK_FILENAME

    if not bank_path.exists():
        raise FileNotFoundError(
            f"No se encuentra `{BANK_FILENAME}` en la raíz del repositorio."
        )

    try:
        df_bank = pd.read_excel(
            bank_path,
            sheet_name="Todas",
            engine="openpyxl",
        )
    except Exception as exc:
        raise RuntimeError(
            "No se pudo abrir el banco de preguntas. "
            "Comprueba que el Excel contiene una hoja llamada `Todas`."
        ) from exc

    required = [
        "ID",
        "Ficha",
        "Tema",
        "Pregunta",
        "Opción A",
        "Opción B",
        "Opción C",
        "Opción D",
        "Correcta",
        "Respuesta correcta",
    ]

    missing = [c for c in required if c not in df_bank.columns]

    if missing:
        raise ValueError(
            "Faltan columnas en el banco: " + ", ".join(missing)
        )

    option_columns = {
        "A": "Opción A",
        "B": "Opción B",
        "C": "Opción C",
        "D": "Opción D",
    }

    bank = []

    for _, row in df_bank.iterrows():
        qid = _safe_text(row["ID"])
        ficha = _safe_text(row["Ficha"])
        tema = _safe_text(row["Tema"])
        question_text = _safe_text(row["Pregunta"])
        correct_letter = _safe_text(row["Correcta"]).upper()

        option_by_letter = {
            letter: _safe_text(row[column])
            for letter, column in option_columns.items()
        }

        options = [
            value for value in option_by_letter.values()
            if value
        ]

        correct_answer = _safe_text(row["Respuesta correcta"])

        if not correct_answer:
            correct_answer = option_by_letter.get(correct_letter, "")

        explanation = (
            _safe_text(row["Explicación"])
            if "Explicación" in df_bank.columns
            else ""
        )

        if (
            qid
            and ficha
            and question_text
            and len(options) >= 2
            and correct_answer
        ):
            bank.append({
                "id": qid,
                "ficha": ficha,
                "tema": tema,
                "dificultad": "",
                "pregunta": question_text,
                "opciones": options,
                "correcta": correct_answer,
                "explicacion": explanation,
            })

    if not bank:
        raise ValueError(
            "El Excel se abrió correctamente, pero no contiene preguntas válidas."
        )

    return bank

def build_submission_csv(student, attempt_id, selected_labels, questions, responses, score):
    output = io.StringIO()
    writer = csv.writer(output, delimiter=";")

    writer.writerow(["ECONOMETRÍA · MINI-TEST"])
    writer.writerow(["ID intento", attempt_id])
    writer.writerow(["Fecha/hora", datetime.now().astimezone().isoformat(timespec="seconds")])
    writer.writerow(["Nombre y apellidos", student["name"]])
    writer.writerow(["DNI", student["dni"]])
    writer.writerow(["NIE", student["nie"]])
    writer.writerow(["Correo UVigo", student["email"]])
    writer.writerow(["Grupo", student["group"]])
    writer.writerow(["Temario", " | ".join(selected_labels)])
    writer.writerow(["Número de preguntas", len(questions)])
    writer.writerow(["Aciertos", score])
    writer.writerow(["Nota sobre 10", round(score / len(questions) * 10, 2)])
    writer.writerow([])

    writer.writerow([
        "N",
        "ID pregunta",
        "Ficha",
        "Tema",
        "Dificultad",
        "Pregunta",
        "Respuesta estudiante",
        "Respuesta correcta",
        "Correcta",
    ])

    for i, (q, response) in enumerate(zip(questions, responses), start=1):
        writer.writerow([
            i,
            q["id"],
            q["ficha"],
            q["tema"],
            q["dificultad"],
            q["pregunta"],
            response or "",
            q["correcta"],
            "Sí" if response == q["correcta"] else "No",
        ])

    return output.getvalue().encode("utf-8-sig")


def send_submission_email(csv_bytes, student, attempt_id, score, n_questions):
    try:
        email_cfg = st.secrets["email"]
        sender = email_cfg["sender"]
        app_password = email_cfg["app_password"]
    except Exception as exc:
        raise RuntimeError(
            "El envío de correo todavía no está configurado en Streamlit Secrets."
        ) from exc

    safe_name = re.sub(r"[^A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ_-]+", "_", student["name"]).strip("_")
    filename = f"econometria_test_{safe_name}_{attempt_id[:8]}.csv"

    msg = EmailMessage()
    msg["Subject"] = (
        f"Econometría · Mini-test · {student['name']} · "
        f"{score}/{n_questions}"
    )
    msg["From"] = sender
    msg["To"] = TEACHER_EMAIL

    msg.set_content(
        "Se ha recibido un nuevo mini-test de Econometría.\n\n"
        f"Estudiante: {student['name']}\n"
        f"DNI: {student['dni']}\n"
        f"NIE: {student['nie']}\n"
        f"Correo UVigo: {student['email']}\n"
        f"Grupo: {student['group'] or 'No indicado'}\n"
        f"Resultado docente: {score}/{n_questions}\n"
        f"ID de intento: {attempt_id}\n\n"
        "El CSV adjunto contiene las preguntas, respuestas del estudiante "
        "y la corrección completa."
    )

    msg.add_attachment(
        csv_bytes,
        maintype="text",
        subtype="csv",
        filename=filename,
    )

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL(
        "smtp.gmail.com",
        465,
        context=context,
        timeout=20,
    ) as smtp:
        smtp.login(sender, app_password)
        smtp.send_message(msg)


def reset_quiz_state():
    for key in [
        "quiz_questions",
        "quiz_attempt_id",
        "quiz_selected_labels",
        "quiz_selected_topics",
        "quiz_submitted",
    ]:
        st.session_state.pop(key, None)




# ============================================================
# CONTROL DEL MODO EXAMEN
# ============================================================

MADRID_TZ = ZoneInfo("Europe/Madrid")
TEST_CONTROL_FILE = Path(tempfile.gettempdir()) / "econometria_test_control.json"


def default_test_control():
    return {
        "enabled": False,
        "start": None,
        "end": None,
        "updated_at": None,
    }


def load_test_control():
    """
    Estado compartido entre las sesiones de la app mientras el contenedor
    de Streamlit siga activo.
    """
    if not TEST_CONTROL_FILE.exists():
        return default_test_control()

    try:
        data = json.loads(
            TEST_CONTROL_FILE.read_text(encoding="utf-8")
        )
    except Exception:
        return default_test_control()

    base = default_test_control()
    base.update(data if isinstance(data, dict) else {})
    return base


def save_test_control(control):
    TEST_CONTROL_FILE.write_text(
        json.dumps(control, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_control_datetime(value):
    if not value:
        return None

    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=MADRID_TZ)
        return dt.astimezone(MADRID_TZ)
    except Exception:
        return None


def get_test_gate_status():
    """
    Devuelve:
      status: 'disabled', 'scheduled', 'open' o 'closed'
      control: configuración
      now: hora actual de Madrid
      start/end: datetimes
    """
    control = load_test_control()
    now = datetime.now(MADRID_TZ)
    start = parse_control_datetime(control.get("start"))
    end = parse_control_datetime(control.get("end"))

    if not control.get("enabled", False):
        status = "disabled"
    elif start is None or end is None:
        status = "disabled"
    elif now < start:
        status = "scheduled"
    elif start <= now <= end:
        status = "open"
    else:
        status = "closed"

    return status, control, now, start, end


def format_madrid_datetime(value):
    if value is None:
        return "—"
    return value.strftime("%d/%m/%Y · %H:%M")


def professor_panel():
    st.sidebar.divider()

    with st.sidebar.expander("🔐 Profesor"):
        st.caption(
            "Control de apertura del Mini-test. "
            "Zona horaria: Europe/Madrid."
        )

        try:
            configured_password = str(
                st.secrets["admin"]["password"]
            )
        except Exception:
            configured_password = ""

        if not configured_password:
            st.warning(
                "Falta configurar `[admin] password` "
                "en los Secrets de Streamlit."
            )
            return

        entered = st.text_input(
            "Contraseña",
            type="password",
            key="professor_password",
        )

        if entered != configured_password:
            if entered:
                st.error("Contraseña incorrecta.")
            return

        st.success("Acceso de profesor")

        status, control, now, start, end = get_test_gate_status()

        status_label = {
            "disabled": "🔴 Desactivado",
            "scheduled": "🟠 Programado",
            "open": "🟢 Abierto",
            "closed": "⚫ Finalizado",
        }[status]

        st.markdown(f"**Estado:** {status_label}")
        st.caption(
            f"Ahora en Madrid: {format_madrid_datetime(now)}"
        )

        default_start = start or now.replace(
            second=0,
            microsecond=0,
        )
        default_end = end or (
            now.replace(second=0, microsecond=0)
            + pd.Timedelta(minutes=60)
        ).to_pydatetime()

        start_date = st.date_input(
            "Fecha de apertura",
            value=default_start.date(),
            key="exam_start_date",
        )

        start_time = st.time_input(
            "Hora de apertura",
            value=default_start.time().replace(
                second=0,
                microsecond=0,
            ),
            step=60,
            key="exam_start_time",
        )

        end_date = st.date_input(
            "Fecha de cierre",
            value=default_end.date(),
            key="exam_end_date",
        )

        end_time = st.time_input(
            "Hora de cierre",
            value=default_end.time().replace(
                second=0,
                microsecond=0,
            ),
            step=60,
            key="exam_end_time",
        )

        if st.button(
            "💾 Programar y activar",
            type="primary",
            use_container_width=True,
        ):
            start_dt = datetime.combine(
                start_date,
                start_time,
                tzinfo=MADRID_TZ,
            )
            end_dt = datetime.combine(
                end_date,
                end_time,
                tzinfo=MADRID_TZ,
            )

            if end_dt <= start_dt:
                st.error(
                    "La hora de cierre debe ser posterior "
                    "a la de apertura."
                )
            else:
                save_test_control({
                    "enabled": True,
                    "start": start_dt.isoformat(),
                    "end": end_dt.isoformat(),
                    "updated_at": datetime.now(
                        MADRID_TZ
                    ).isoformat(),
                })
                st.success("Mini-test programado.")
                st.rerun()

        c1, c2 = st.columns(2)

        with c1:
            if st.button(
                "🟢 Abrir ahora",
                use_container_width=True,
            ):
                end_dt = end or (
                    now + pd.Timedelta(minutes=60)
                ).to_pydatetime()

                if end_dt <= now:
                    end_dt = (
                        now + pd.Timedelta(minutes=60)
                    ).to_pydatetime()

                save_test_control({
                    "enabled": True,
                    "start": now.isoformat(),
                    "end": end_dt.isoformat(),
                    "updated_at": now.isoformat(),
                })
                st.rerun()

        with c2:
            if st.button(
                "🔴 Cerrar ahora",
                use_container_width=True,
            ):
                save_test_control({
                    "enabled": False,
                    "start": control.get("start"),
                    "end": control.get("end"),
                    "updated_at": now.isoformat(),
                })
                st.rerun()

        if start and end:
            st.caption(
                "Apertura: "
                f"{format_madrid_datetime(start)}\n\n"
                "Cierre: "
                f"{format_madrid_datetime(end)}"
            )

        st.warning(
            "La programación se conserva mientras la instancia "
            "de Streamlit permanezca activa. Si la app se reinicia "
            "completamente, vuelve a programar el examen."
        )


def enforce_test_gate_for_students():
    """
    Bloquea el Mini-test antes de cargar el banco o mostrar preguntas.
    """
    status, control, now, start, end = get_test_gate_status()

    if status == "open":
        st.success(
            "🟢 Mini-test abierto. "
            f"Cierra a las {format_madrid_datetime(end)}."
        )
        return True

    if status == "scheduled":
        st.info(
            "🔒 El Mini-test todavía no está abierto."
        )
        st.markdown(
            f"**Apertura:** {format_madrid_datetime(start)}  \n"
            f"**Cierre:** {format_madrid_datetime(end)}"
        )
        st.caption(
            f"Hora actual: {format_madrid_datetime(now)}"
        )
        return False

    if status == "closed":
        st.warning(
            "🔒 El plazo del Mini-test ha finalizado."
        )
        if end:
            st.caption(
                f"El test cerró el {format_madrid_datetime(end)}."
            )
        return False

    st.info(
        "🔒 El Mini-test no está disponible en este momento."
    )
    st.caption(
        "El profesor activará la prueba cuando corresponda."
    )
    return False


# ============================================================
# NAVEGACIÓN
# ============================================================

st.sidebar.markdown("## 📈 Econometría")
st.sidebar.caption("Universidade de Vigo")

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
    ],
)

st.sidebar.divider()
st.sidebar.caption(
    "Material complementario de clase · "
    "Prof. Alessandro Indelicato"
)

professor_panel()


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
    c4.metric("Datos", "CSV / XLSX")

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
        "Sube la base de datos con la que vas a trabajar en clase."
    )

    df, source = require_uploaded_dataset(
        key="lab_dataset",
        label="Sube la base de datos del ejercicio",
    )

    st.caption(
        f"Archivo activo: **{source}** · "
        f"{len(df)} observaciones · {df.shape[1]} variables"
    )

    st.dataframe(df.head(12), use_container_width=True)

    num = numeric_columns(df)

    if len(num) < 2:
        st.error("La base necesita al menos dos variables numéricas.")
        st.stop()

    default_y = 0

    y = st.selectbox(
        "Variable dependiente (Y)",
        num,
        index=default_y
    )

    x_candidates = [c for c in num if c != y]

    defaults = x_candidates[: min(2, len(x_candidates))]

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

    df, source = require_uploaded_dataset(
        key="contrast_dataset",
        label="Sube la base de datos para el contraste",
    )
    st.caption(f"Archivo activo: **{source}**")
    num = numeric_columns(df)

    default_y = 0

    y = st.selectbox(
        "Y",
        num,
        index=default_y,
        key="contrast_y",
    )

    xs_all = [c for c in num if c != y]

    defaults = xs_all[: min(2, len(xs_all))]

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

    # Acceso por nombre de variable: evita KeyError con pandas recientes
    b = float(model.params[coef])
    se = float(model.bse[coef])
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

    df, source = require_uploaded_dataset(
        key="diag_dataset",
        label="Sube la base de datos para el diagnóstico",
    )
    st.caption(f"Archivo activo: **{source}**")
    num = numeric_columns(df)

    default_y = 0

    y = st.selectbox(
        "Y",
        num,
        index=default_y,
        key="diag_y",
    )

    xs_all = [c for c in num if c != y]

    defaults = xs_all[: min(2, len(xs_all))]

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

    df, source = require_uploaded_dataset(
        key="gretl_dataset",
        label="Sube la base de datos para generar el script",
    )
    st.caption(f"Archivo activo: **{source}**")
    num = numeric_columns(df)

    default_y = 0

    y = st.selectbox(
        "Variable dependiente",
        num,
        index=default_y,
        key="gretl_y",
    )

    xs_all = [c for c in num if c != y]

    defaults = xs_all[: min(2, len(xs_all))]

    xs = st.multiselect(
        "Explicativas",
        xs_all,
        default=defaults[:4] or xs_all[:2],
        key="gretl_x",
    )

    if xs:
        script = gretl_script(y, xs, filename=source)

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
        "Selecciona el temario y realiza una prueba aleatoria de hasta 10 preguntas. "
        "La prueba se envía al profesor; no se muestra la calificación ni las soluciones."
    )

    # El banco y las preguntas solo se cargan cuando el examen está abierto.
    if not enforce_test_gate_for_students():
        st.stop()

    try:
        bank = load_question_bank()
    except Exception as exc:
        st.error("No se pudo cargar el banco de preguntas.")
        st.code(str(exc), language="text")
        st.info(
            f"Debe existir `{BANK_FILENAME}` en la raíz del repositorio "
            "y contener la hoja `Todas`."
        )
        st.stop()

    st.caption(
        f"Banco cargado: {len(bank):,} preguntas disponibles."
        .replace(",", ".")
    )

    # --------------------------------------------------------
    # Identificación del estudiante
    # --------------------------------------------------------

    st.subheader("1. Identificación")

    col1, col2, col3 = st.columns(3)

    with col1:
        student_name = st.text_input(
            "Nombre y apellidos *",
            placeholder="Nombre Apellido1 Apellido2",
        )

    with col2:
        student_dni = st.text_input(
            "DNI *",
            placeholder="Ej.: 12345678Z",
        )

    with col3:
        student_nie = st.text_input(
            "NIE *",
            placeholder="Introduce tu NIE",
        )

    col4, col5 = st.columns(2)

    with col4:
        student_email = st.text_input(
            "Correo UVigo *",
            placeholder="usuario@uvigo.gal",
        )

    with col5:
        student_group = st.text_input(
            "Grupo",
            placeholder="Ej.: ADE B4",
        )

    # --------------------------------------------------------
    # Selección de temario
    # --------------------------------------------------------

    st.subheader("2. Temario del test")

    selected_labels = st.multiselect(
        "Selecciona una o varias partes del temario",
        options=list(FICHA_LABELS.values()),
        default=[],
        disabled="quiz_questions" in st.session_state,
    )

    selected_fichas = [
        LABEL_TO_FICHA[label]
        for label in selected_labels
    ]

    topic_pool = sorted({
        q["tema"]
        for q in bank
        if q["ficha"] in selected_fichas and q["tema"]
    })

    selected_topics = st.multiselect(
        "Temas específicos dentro de esas fichas (opcional)",
        options=topic_pool,
        default=[],
        help="Si lo dejas vacío, entran todos los temas de las fichas seleccionadas.",
        disabled="quiz_questions" in st.session_state,
    )

    n_questions = st.slider(
        "Número de preguntas",
        min_value=1,
        max_value=10,
        value=10,
        disabled="quiz_questions" in st.session_state,
    )

    if "quiz_questions" not in st.session_state:

        if st.button(
            "🎲 Generar test aleatorio",
            type="primary",
            use_container_width=True,
        ):

            if not selected_fichas:
                st.warning("Selecciona al menos una parte del temario.")

            elif (
                not student_name.strip()
                or not student_dni.strip()
                or not student_nie.strip()
                or not student_email.strip()
            ):
                st.warning(
                    "Introduce nombre y apellidos, DNI, NIE y correo UVigo antes de generar la prueba."
                )

            else:
                candidates = [
                    q for q in bank
                    if q["ficha"] in selected_fichas
                    and (
                        not selected_topics
                        or q["tema"] in selected_topics
                    )
                ]

                if not candidates:
                    st.warning(
                        "No hay preguntas disponibles con esa combinación de temario."
                    )

                else:
                    k = min(n_questions, len(candidates))
                    chosen = random.SystemRandom().sample(candidates, k)

                    frozen_questions = []

                    for q in chosen:
                        qcopy = dict(q)
                        shuffled = list(qcopy["opciones"])
                        random.SystemRandom().shuffle(shuffled)
                        qcopy["opciones"] = shuffled
                        frozen_questions.append(qcopy)

                    st.session_state.quiz_questions = frozen_questions
                    st.session_state.quiz_attempt_id = str(uuid.uuid4())
                    st.session_state.quiz_selected_labels = selected_labels
                    st.session_state.quiz_selected_topics = selected_topics
                    st.session_state.quiz_submitted = False
                    st.rerun()

    # --------------------------------------------------------
    # Test congelado
    # --------------------------------------------------------

    if "quiz_questions" in st.session_state:

        questions = st.session_state.quiz_questions
        attempt_id = st.session_state.quiz_attempt_id

        st.divider()
        st.subheader("3. Prueba")

        st.info(
            f"Se han seleccionado {len(questions)} preguntas al azar. "
            "Las preguntas y el orden de las respuestas permanecen fijos durante este intento."
        )

        if st.session_state.get("quiz_submitted", False):
            st.success(
                "✅ Prueba enviada correctamente al profesor."
            )
            st.write(
                "No se muestra la puntuación ni las respuestas correctas."
            )

            if st.button("Hacer un nuevo test"):
                reset_quiz_state()
                st.rerun()

        else:
            with st.form("student_quiz_form"):
                responses = []

                for i, q in enumerate(questions, start=1):
                    st.markdown(f"### Pregunta {i}")
                    st.write(q["pregunta"])

                    response = st.radio(
                        "Selecciona una respuesta",
                        q["opciones"],
                        key=f"{attempt_id}_q_{i}",
                        index=None,
                    )
                    responses.append(response)

                    st.divider()

                accept = st.checkbox(
                    "Confirmo que quiero enviar este intento al profesor."
                )

                submitted = st.form_submit_button(
                    "📨 Enviar prueba",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:

                if not accept:
                    st.warning(
                        "Marca la casilla de confirmación antes de enviar."
                    )

                elif any(r is None for r in responses):
                    st.warning(
                        "Debes responder todas las preguntas antes de enviar."
                    )

                else:
                    score = sum(
                        response == q["correcta"]
                        for q, response in zip(questions, responses)
                    )

                    student = {
                        "name": student_name.strip(),
                        "dni": student_dni.strip().upper(),
                        "nie": student_nie.strip().upper(),
                        "email": student_email.strip(),
                        "group": student_group.strip(),
                    }

                    csv_bytes = build_submission_csv(
                        student=student,
                        attempt_id=attempt_id,
                        selected_labels=st.session_state.quiz_selected_labels,
                        questions=questions,
                        responses=responses,
                        score=score,
                    )

                    try:
                        send_submission_email(
                            csv_bytes=csv_bytes,
                            student=student,
                            attempt_id=attempt_id,
                            score=score,
                            n_questions=len(questions),
                        )

                    except Exception:
                        st.error(
                            "No se pudo enviar la prueba. "
                            "No cierres la página y avisa al profesor."
                        )

                    else:
                        st.session_state.quiz_submitted = True
                        st.rerun()
