# Econometría Interactiva · UVigo

## Archivos de la raíz del repositorio

```text
app.py
requirements.txt
MooVi_Econometria_Ficha00_Ficha06_1750.xml
README.md
.gitignore
.streamlit/
    config.toml
```

## Mini-test

- El alumno selecciona una o varias fichas del temario.
- Puede afinar por temas concretos.
- Se generan entre 1 y 10 preguntas aleatorias.
- Las opciones también se barajan.
- La selección queda congelada durante el intento.
- El alumno no ve nota, porcentaje, soluciones ni feedback de corrección.
- Al enviar, se genera un CSV con identificación, preguntas, respuestas y corrección.
- El CSV se envía automáticamente a `alessandro.indelicato.a@gmail.com`.

## Banco de preguntas

Sube a la raíz del repositorio el XML conjunto:

`MooVi_Econometria_Ficha00_Ficha06_1750.xml`

La app lo lee directamente y utiliza sus Ficha00–Ficha06, temas, dificultad,
opciones y respuesta correcta.

## Configurar el envío por correo

NO escribas la contraseña en `app.py` y NO subas `.streamlit/secrets.toml` a GitHub.

En Streamlit Community Cloud:

1. Abre tu aplicación.
2. Entra en **App settings / Settings**.
3. Abre **Secrets**.
4. Añade:

```toml
[email]
sender = "TU_CUENTA_GMAIL@gmail.com"
app_password = "CONTRASENA_DE_APLICACION_DE_GOOGLE"
```

El destinatario está fijado en la aplicación como:

`alessandro.indelicato.a@gmail.com`

## Despliegue

Main file path:

`app.py`
