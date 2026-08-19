# Econometría Interactiva

Aplicación docente en **Streamlit** para acompañar las clases de Econometría de ADE.

## Qué incluye

- Mapa del curso en 6 bloques.
- Laboratorio interactivo de regresión lineal por MCO.
- Interpretación de coeficientes y formas funcionales.
- Contrastes `t` e intervalos de confianza.
- Diagnóstico: VIF, Breusch–Pagan, White, Jarque–Bera, RESET y Durbin–Watson.
- Generador de comandos básicos para Gretl.
- Mini-test autocorregible.
- Guía y checklist del microproyecto.
- Dataset sintético de ADE para practicar sin depender de una base externa.

## Estructura

```text
econometria_streamlit/
├── app.py
├── common.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── data/
│   └── ventas_ade_demo.csv
└── views/
    ├── inicio.py
    ├── temario.py
    ├── laboratorio.py
    ├── interpretacion.py
    ├── contrastes.py
    ├── diagnostico.py
    ├── gretl.py
    ├── quiz.py
    └── microproyecto.py
```

## Ejecutar en local

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Instala dependencias y ejecuta:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar en GitHub

1. Crea un repositorio nuevo en GitHub, por ejemplo `econometria-interactiva`.
2. Sube el contenido de esta carpeta al repositorio.
3. En Streamlit Community Cloud, crea una nueva app.
4. Selecciona el repositorio, la rama y `app.py` como archivo de entrada.
5. Pulsa **Deploy**.

## Nota docente

La app está pensada como complemento de Gretl, no como sustituto. El objetivo es que cada resultado termine en una interpretación económica o estadística y que el alumnado distinga ajuste, inferencia y causalidad.
