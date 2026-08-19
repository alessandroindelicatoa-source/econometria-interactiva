# Econometría Interactiva · UVigo

## Estructura que debes subir a la raíz del repositorio

```text
app.py
requirements.txt
Banco_preguntas_MooVi_Econometria.xlsx
README.md
.gitignore
.streamlit/
    config.toml
```

## Mini-test

- Banco real: 1.750 preguntas, 250 por Ficha00–Ficha06.
- El alumno introduce nombre, correo UVigo y grupo.
- Puede seleccionar una o varias fichas del temario.
- Puede restringir además a temas concretos.
- Se generan entre 1 y 10 preguntas aleatorias.
- Las opciones A–D también se barajan.
- El intento queda congelado durante la prueba.
- El alumno no ve nota, porcentaje, respuestas correctas ni explicación.
- Al enviar, se genera un CSV con la corrección completa para el profesor.
- El correo de destino es `alessandro.indelicato.a@gmail.com`.

## Banco de preguntas

La aplicación lee directamente:

`Banco_preguntas_MooVi_Econometria.xlsx`

y concretamente la hoja **Todas**, con estas columnas:

`ID`, `Ficha`, `Tema`, `Pregunta`, `Opción A`, `Opción B`,
`Opción C`, `Opción D`, `Correcta`, `Respuesta correcta`, `Explicación`.

No hace falta ningún XML.

## Envío por correo

No pongas contraseñas en GitHub.

En Streamlit Community Cloud abre la configuración de la app y añade en
**Secrets**:

```toml
[email]
sender = "TU_CUENTA_GMAIL@gmail.com"
app_password = "CONTRASENA_DE_APLICACION_DE_GOOGLE"
```

El destinatario está fijado en `app.py` como:

`alessandro.indelicato.a@gmail.com`

## Despliegue

En Streamlit Cloud:

- Branch: `main`
- Main file path: `app.py`

Si ya tenías una versión desplegada, sustituye los archivos, haz Commit y
después reinicia la aplicación.


## Datos de los ejercicios

La aplicación NO incluye ningún dataset demo.

Para usar:

- Laboratorio MCO
- Contrastes
- Diagnóstico
- Generador Gretl

el estudiante debe subir su propia base en formato `CSV` o `XLSX`.

El único Excel incluido en el repositorio es
`Banco_preguntas_MooVi_Econometria.xlsx`, que contiene el banco docente
de 1.750 preguntas del Mini-test y no es una base de datos demo.
