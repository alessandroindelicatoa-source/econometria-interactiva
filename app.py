from pathlib import Path
import zipfile, shutil, py_compile, re

src_zip = Path("/mnt/data/econometria_interactiva_V6_LISTO_GITHUB.zip")
work = Path("/mnt/data/econometria_interactiva_V7_DNI_NIE")

if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True)

with zipfile.ZipFile(src_zip, "r") as z:
    z.extractall(work)

app_path = work / "app.py"
text = app_path.read_text(encoding="utf-8")

# 1) Añadir DNI/NIE en identificación
old_ident = '''    col1, col2 = st.columns(2)

    with col1:
        student_name = st.text_input(
            "Nombre y apellidos *",
            placeholder="Nombre Apellido1 Apellido2",
        )

    with col2:
        student_email = st.text_input(
            "Correo UVigo *",
            placeholder="usuario@uvigo.gal",
        )

    student_group = st.text_input(
        "Grupo",
        placeholder="Ej.: ADE B4",
    )
'''

new_ident = '''    col1, col2 = st.columns(2)

    with col1:
        student_name = st.text_input(
            "Nombre y apellidos *",
            placeholder="Nombre Apellido1 Apellido2",
        )

    with col2:
        student_id = st.text_input(
            "DNI/NIE *",
            placeholder="Ej.: 12345678Z / X1234567L",
        )

    col3, col4 = st.columns(2)

    with col3:
        student_email = st.text_input(
            "Correo UVigo *",
            placeholder="usuario@uvigo.gal",
        )

    with col4:
        student_group = st.text_input(
            "Grupo",
            placeholder="Ej.: ADE B4",
        )
'''

if old_ident not in text:
    raise RuntimeError("No se encontró el bloque de identificación esperado.")
text = text.replace(old_ident, new_ident, 1)

# 2) Hacer obligatorio DNI/NIE al generar test
old_validation = '''            elif not student_name.strip() or not student_email.strip():
                st.warning("Introduce nombre y correo UVigo antes de generar la prueba.")
'''

new_validation = '''            elif (
                not student_name.strip()
                or not student_id.strip()
                or not student_email.strip()
            ):
                st.warning(
                    "Introduce nombre y apellidos, DNI/NIE y correo UVigo antes de generar la prueba."
                )
'''

if old_validation not in text:
    raise RuntimeError("No se encontró la validación de identificación.")
text = text.replace(old_validation, new_validation, 1)

# 3) Añadir DNI/NIE al diccionario del estudiante
old_student = '''                    student = {
                        "name": student_name.strip(),
                        "email": student_email.strip(),
                        "group": student_group.strip(),
                    }
'''

new_student = '''                    student = {
                        "name": student_name.strip(),
                        "id_number": student_id.strip().upper(),
                        "email": student_email.strip(),
                        "group": student_group.strip(),
                    }
'''

if old_student not in text:
    raise RuntimeError("No se encontró el diccionario del estudiante.")
text = text.replace(old_student, new_student, 1)

# 4) Añadir DNI/NIE al CSV
old_csv = '''    writer.writerow(["Nombre y apellidos", student["name"]])
    writer.writerow(["Correo UVigo", student["email"]])
    writer.writerow(["Grupo", student["group"]])
'''

new_csv = '''    writer.writerow(["Nombre y apellidos", student["name"]])
    writer.writerow(["DNI/NIE", student["id_number"]])
    writer.writerow(["Correo UVigo", student["email"]])
    writer.writerow(["Grupo", student["group"]])
'''

if old_csv not in text:
    raise RuntimeError("No se encontró el bloque de cabecera CSV.")
text = text.replace(old_csv, new_csv, 1)

# 5) Añadir DNI/NIE al cuerpo del email
old_mail = '''        f"Estudiante: {student['name']}\\n"
        f"Correo UVigo: {student['email']}\\n"
        f"Grupo: {student['group'] or 'No indicado'}\\n"
'''

new_mail = '''        f"Estudiante: {student['name']}\\n"
        f"DNI/NIE: {student['id_number']}\\n"
        f"Correo UVigo: {student['email']}\\n"
        f"Grupo: {student['group'] or 'No indicado'}\\n"
'''

if old_mail not in text:
    raise RuntimeError("No se encontró el cuerpo del email.")
text = text.replace(old_mail, new_mail, 1)

# 6) Validación de sintaxis
app_path.write_text(text, encoding="utf-8")
py_compile.compile(str(app_path), doraise=True)

# 7) Crear ZIP final
out_zip = Path("/mnt/data/econometria_interactiva_V7_DNI_NIE.zip")
if out_zip.exists():
    out_zip.unlink()

with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for p in work.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts:
            z.write(p, p.relative_to(work))

print("✅ Campo DNI/NIE añadido y obligatorio.")
print("✅ DNI/NIE incluido en CSV y correo al profesor.")
print("✅ Sintaxis validada.")
print(f"📦 ZIP: {out_zip}")
