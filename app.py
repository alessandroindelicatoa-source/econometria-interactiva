from pathlib import Path
import re, py_compile

src = Path("/mnt/data/app.py")
dst = Path("/mnt/data/app_DNI_y_NIE.py")

text = src.read_text(encoding="utf-8")

# 1) Replace single DNI/NIE field with two separate fields
old = '''    col1, col2 = st.columns(2)

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

new = '''    col1, col2, col3 = st.columns(3)

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
'''

if old not in text:
    raise RuntimeError("No se encontró el bloque de identificación esperado.")
text = text.replace(old, new, 1)

# 2) Validation
old = '''            elif (
                not student_name.strip()
                or not student_id.strip()
                or not student_email.strip()
            ):
                st.warning(
                    "Introduce nombre y apellidos, DNI/NIE y correo UVigo antes de generar la prueba."
                )
'''
new = '''            elif (
                not student_name.strip()
                or not student_dni.strip()
                or not student_nie.strip()
                or not student_email.strip()
            ):
                st.warning(
                    "Introduce nombre y apellidos, DNI, NIE y correo UVigo antes de generar la prueba."
                )
'''
if old not in text:
    raise RuntimeError("No se encontró la validación de identificación.")
text = text.replace(old, new, 1)

# 3) Student dict
old = '''                    student = {
                        "name": student_name.strip(),
                        "id_number": student_id.strip().upper(),
                        "email": student_email.strip(),
                        "group": student_group.strip(),
                    }
'''
new = '''                    student = {
                        "name": student_name.strip(),
                        "dni": student_dni.strip().upper(),
                        "nie": student_nie.strip().upper(),
                        "email": student_email.strip(),
                        "group": student_group.strip(),
                    }
'''
if old not in text:
    raise RuntimeError("No se encontró el diccionario student.")
text = text.replace(old, new, 1)

# 4) CSV header rows
old = '''    writer.writerow(["Nombre y apellidos", student["name"]])
    writer.writerow(["DNI/NIE", student["id_number"]])
    writer.writerow(["Correo UVigo", student["email"]])
    writer.writerow(["Grupo", student["group"]])
'''
new = '''    writer.writerow(["Nombre y apellidos", student["name"]])
    writer.writerow(["DNI", student["dni"]])
    writer.writerow(["NIE", student["nie"]])
    writer.writerow(["Correo UVigo", student["email"]])
    writer.writerow(["Grupo", student["group"]])
'''
if old not in text:
    raise RuntimeError("No se encontró el bloque CSV.")
text = text.replace(old, new, 1)

# 5) Email body
old = '''        f"Estudiante: {student['name']}\\n"
        f"DNI/NIE: {student['id_number']}\\n"
        f"Correo UVigo: {student['email']}\\n"
'''
new = '''        f"Estudiante: {student['name']}\\n"
        f"DNI: {student['dni']}\\n"
        f"NIE: {student['nie']}\\n"
        f"Correo UVigo: {student['email']}\\n"
'''
if old not in text:
    raise RuntimeError("No se encontró el cuerpo de email.")
text = text.replace(old, new, 1)

dst.write_text(text, encoding="utf-8")
py_compile.compile(str(dst), doraise=True)

print("✅ DNI y NIE separados.")
print("✅ Ambos son obligatorios.")
print("✅ Ambos se guardan en CSV y correo.")
print("✅ Sintaxis validada.")
print(dst)
