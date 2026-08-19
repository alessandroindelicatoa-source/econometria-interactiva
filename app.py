import streamlit as st

st.set_page_config(
    page_title="Econometría Interactiva · UVigo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
  --main: #12355B;
  --accent: #1F6E8C;
  --soft: #F3F7FA;
}
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1250px;}
[data-testid="stSidebar"] {border-right: 1px solid #E6ECF1;}
h1, h2, h3 {letter-spacing: -0.02em;}
.hero {
  padding: 1.3rem 1.5rem;
  border-radius: 18px;
  background: linear-gradient(135deg, #12355B 0%, #1F6E8C 100%);
  color: white;
  margin-bottom: 1rem;
}
.hero h1 {color:white; margin:0; font-size:2.15rem;}
.hero p {margin: .45rem 0 0; opacity:.93;}
.card {
  border: 1px solid #E1E8EE;
  border-radius: 14px;
  padding: 1rem 1.1rem;
  background: white;
  min-height: 130px;
}
.kicker {font-size:.82rem; text-transform:uppercase; letter-spacing:.08em; color:#557;}
.small-note {font-size:.9rem; color:#5B6573;}
</style>
""", unsafe_allow_html=True)

pages = {
    "Curso": [
        st.Page("views/inicio.py", title="Inicio", icon="🏠", default=True),
        st.Page("views/temario.py", title="Temario y sesiones", icon="🧭"),
    ],
    "Laboratorio": [
        st.Page("views/laboratorio.py", title="Regresión MCO", icon="📈"),
        st.Page("views/interpretacion.py", title="Interpretar coeficientes", icon="🔎"),
        st.Page("views/contrastes.py", title="Contrastes e IC", icon="🧪"),
        st.Page("views/diagnostico.py", title="Diagnóstico", icon="🩺"),
        st.Page("views/gretl.py", title="Generador Gretl", icon="⌨️"),
    ],
    "Evaluación": [
        st.Page("views/quiz.py", title="Mini-test", icon="✅"),
        st.Page("views/microproyecto.py", title="Microproyecto", icon="🗂️"),
    ],
}

pg = st.navigation(pages)
pg.run()
