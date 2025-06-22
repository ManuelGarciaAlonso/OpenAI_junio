# Importación de bibliotecas necesarias
import os
import time
import fitz  # PyMuPDF (requerido por pymupdf4llm)
from pathlib import Path
from pymupdf4llm import to_markdown # Conversor de PDF a markdown
import subprocess
import shutil
import re
import tempfile # Para crear directorios temporales seguros
import traceback # Para imprimir stack traces completos
import base64 # Necesario si quisiéramos decodificar, pero ahora solo eliminamos

# === CONFIGURACIÓN ===
# Rutas a los directorios que contienen los PDFs
BASE_PDF_DIR = Path("/mnt/c/Users/34644/Desktop/TFM/FS") # Ruta base común
THEORY_DIR = BASE_PDF_DIR / "Teoria"
EXERCISES_DIR = BASE_PDF_DIR / "BoletinesEjercicios"
PRACTICES_DIR = BASE_PDF_DIR / "Prácticas"

# Lista de directorios a escanear y sus categorías
DIRS_TO_SCAN = [
    (THEORY_DIR, "Teoría"),
    (EXERCISES_DIR, "Ejercicios"),
    (PRACTICES_DIR, "Prácticas"),
]

# Directorio de salida para guardar los archivos procesados individualmente
OUTPUT_DIR = Path("/mnt/c/Users/34644/Desktop/TFM/Ollama_marzo/Processed_Texts/preprocessed_markdown")

# === FUNCIONES ===
def generate_output_filename(pdf_path, category):
    """Genera un nombre de archivo adecuado para el archivo de salida"""
    # Eliminar la extensión .pdf y reemplazar espacios o caracteres problemáticos
    pdf_name_base = pdf_path.stem.replace(' ', '_')
    # Formato: [Categoria]_[NombreOriginalPDF].md
    return f"{category}_{pdf_name_base}.txt"

def find_pdfs_in_dirs(dirs_to_scan):
    """
    Busca archivos .pdf en los directorios especificados.
    Devuelve una lista de tuplas: [(Path(ruta_pdf), "Categoria"), ...]
    """
    all_pdfs = []
    print("🔍 Escaneando directorios en busca de archivos PDF...")
    for dir_path, category in dirs_to_scan:
        if not dir_path.is_dir():
            print(f"  ⚠️ Advertencia: El directorio '{dir_path}' no existe o no es accesible. Saltando categoría '{category}'.")
            continue

        print(f"  📂 Escaneando '{dir_path}' (Categoría: {category})")
        found_files = list(dir_path.glob('*.pdf'))
        if not found_files:
             print(f"    -> No se encontraron archivos .pdf en este directorio.")
        else:
            print(f"    -> Encontrados {len(found_files)} archivo(s) .pdf.")
            for pdf_file in found_files:
                all_pdfs.append((pdf_file, category))
    print(f"✨ Escaneo completo. Total de PDFs encontrados: {len(all_pdfs)}")
    return all_pdfs

def extraer_texto_docling_primero(pdf_path_obj):
    """
    Intenta extraer texto usando Docling. Si falla, usa PyMuPDF4LLM.
    Recibe un objeto Path.
    Devuelve el texto extraído (Markdown) y el método utilizado.
    """
    pdf_path_str = str(pdf_path_obj) # Convertir a string para comandos y algunas libs
    print(f"  Intentando extracción con Docling...")
    docling_success = False
    texto_extraido = ""
    metodo_usado = "Ninguno"
    docling_stderr_output = "No disponible" # Inicializar stderr

    # Crear un directorio temporal para la salida de Docling
    with tempfile.TemporaryDirectory() as docling_output_dir:
        output_dir_obj = Path(docling_output_dir)
        try:
            # Asegurarse de que las rutas se pasan entre comillas al shell
            docling_cmd = f'docling "{pdf_path_str}" --to md --output "{output_dir_obj}"'
            print(f"    Ejecutando comando: {docling_cmd}")

            result = subprocess.run(docling_cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore', timeout=180) # Timeout de 3 minutos
            docling_stderr_output = result.stderr # Guardar stderr para posible mensaje de error

            # Comprobar si Docling generó archivos .md
            md_files_generated = list(output_dir_obj.glob("*.md"))

            if result.returncode == 0 and md_files_generated:
                combined = ""
                for md_file in sorted(md_files_generated):
                    try:
                        with open(md_file, "r", encoding="utf-8") as f:
                            combined += f.read() + "\n\n" # Añadir espacio extra
                    except Exception as read_err:
                        print(f"    WARN: No se pudo leer el archivo temporal {md_file.name}: {read_err}")
                        continue

                if combined.strip():
                    texto_extraido = combined.strip()
                    metodo_usado = "Docling"
                    docling_success = True
                    print("  ✅ Extracción con Docling exitosa.")
                else:
                    print("  ⚠️ Docling terminó pero no se encontró contenido en los archivos .md.")

            else:
                print(f"  ⚠️ Docling falló o no generó archivos .md (Código de retorno: {result.returncode}).")


        except subprocess.TimeoutExpired:
             print("  ❌ Error: Docling excedió el tiempo límite.")
             docling_stderr_output = "TimeoutExpired"
        except Exception as e:
            print(f"  ❌ Error inesperado durante la ejecución de Docling: {e}")
            docling_stderr_output = str(e)

    # --- Fallback a PyMuPDF4LLM si Docling falló ---
    if not docling_success:
        print(f"  Intentando extracción con PyMuPDF4LLM (fallback)...")
        try:
            # Nota: PyMuPDF4LLM puede requerir la ruta como string
            texto_extraido = to_markdown(pdf_path_str)
            metodo_usado = "PyMuPDF4LLM"
            print("  ✅ Extracción con PyMuPDF4LLM exitosa.")
        except Exception as e:
            print(f"  ❌ Error: PyMuPDF4LLM también falló: {e}")
            texto_extraido = (f"ERROR: No se pudo extraer texto.\n"
                              f"Docling falló (stderr):\n{docling_stderr_output}\n\n"
                              f"PyMuPDF4LLM falló (error):\n{e}")
            metodo_usado = "Fallo_Total"

    return texto_extraido, metodo_usado

def quitar_imagenes_base64_markdown(texto_markdown):
    """Elimina las etiquetas de imagen Markdown con datos base64."""
    # Regex para encontrar ![...](data:image/...)
    patron_imagen_base64 = r"!\[.*?\]\(data:image\/[a-zA-Z0-9\+\/]+;base64,[a-zA-Z0-9\+\/=\s\n]+\)"
    # Contar cuántas se van a eliminar
    num_imagenes = len(re.findall(patron_imagen_base64, texto_markdown))
    # Reemplazar con string vacío
    texto_sin_imagenes = re.sub(patron_imagen_base64, '', texto_markdown)
    if num_imagenes > 0:
        print(f"  🖼️ Se eliminaron {num_imagenes} etiquetas de imagen base64 del Markdown.")
    return texto_sin_imagenes

def parsear_ejercicios_v2(texto_markdown):
    """
    Parsea el texto Markdown de boletines de ejercicios para extraerlos individualmente.
    Busca patrones como "- 1." o "- (1)" al inicio de línea.
    Devuelve una lista de diccionarios: [{'numero': str, 'texto': str}]
    """
    # Patrón mejorado para detectar "- <num>." o "- (<num>)" al inicio de línea.
    # Usa un lookahead (?=...) para dividir *antes* del patrón, manteniendo el patrón al inicio.
    patron_split = r"(?=\s*-\s+(?:\d+\.|\(\d+\))\s+)"
    # Dividimos el texto en chunks basados en el patrón. El primer chunk puede ser texto introductorio.
    chunks = re.split(patron_split, texto_markdown, flags=re.MULTILINE)

    ejercicios = []
    texto_introductorio = ""

    # Patrón para extraer el número y el inicio del texto del ejercicio (ahora que está al inicio del chunk)
    # Se asume que el patrón de inicio ahora está al comienzo del chunk
    patron_extract = re.compile(r"^\s*-\s+(?:(\d+)\.|\((\d+)\))\s+(.*)", re.DOTALL)
    # re.DOTALL para que '.' capture también saltos de línea

    if chunks:
        # El primer chunk podría no ser un ejercicio si hay texto antes del primero
        # (como encabezados o introducciones que no queremos en el primer ejercicio)
        primer_match = patron_extract.match(chunks[0])
        if not primer_match and chunks[0].strip():
             # Si el primer chunk no empieza como ejercicio y no está vacío, es texto introductorio
             texto_introductorio = chunks[0].strip()
             # Procesamos desde el segundo chunk
             chunks_a_procesar = chunks[1:]
        else:
             # El primer chunk ya es un ejercicio o está vacío
             chunks_a_procesar = chunks
    else:
        chunks_a_procesar = []


    for chunk in chunks_a_procesar:
        # Intentamos extraer el número y el texto de este chunk
        match = patron_extract.match(chunk)
        if match:
            # Captura el número del Grupo 1 si es formato "- <num>." o del Grupo 2 si es "- (<num>)"
            numero_str = match.group(1) if match.group(1) else match.group(2)
            # El texto del ejercicio es todo lo que sigue al patrón de inicio dentro del chunk
            texto_ejercicio_completo = chunk.strip() # Usar todo el chunk limpio

            if numero_str and texto_ejercicio_completo: # Asegurarse de que tenemos ambos
                ejercicios.append({
                    "numero": numero_str,
                    "texto": texto_ejercicio_completo
                })
        elif chunk.strip():
            # Si un chunk no coincide con el patrón de ejercicio pero no está vacío,
            # podría ser parte del ejercicio anterior o contenido intermedio.
            pass

    print(f"  🧩 Parseo v2 completado. Detectados {len(ejercicios)} ejercicios.")
    if texto_introductorio:
        print(f"  ℹ️ Se encontró texto introductorio antes del primer ejercicio.")

    return ejercicios

# === MAIN ===
if __name__ == "__main__":
    # Crear el directorio de salida si no existe
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Iniciando preprocesamiento de PDFs...")
    print(f"💾 Directorio de salida: {OUTPUT_DIR}")

    # 1. Encontrar todos los PDFs a procesar
    pdfs_a_procesar = find_pdfs_in_dirs(DIRS_TO_SCAN)

    if not pdfs_a_procesar:
        print("\n❌ No se encontraron archivos PDF en los directorios especificados. Saliendo.")
        exit()

    print(f"\n⏱️ Iniciando procesamiento de {len(pdfs_a_procesar)} archivo(s) PDF...")

    # Iterar sobre los PDFs encontrados
    for pdf_path, category in pdfs_a_procesar:
        pdf_name = pdf_path.name # Nombre del archivo con extensión
        
        # Generar nombre de archivo de salida para este PDF
        output_filename = generate_output_filename(pdf_path, category)
        output_filepath = OUTPUT_DIR / output_filename
        
        print(f"\n{'='*60}")
        print(f"📄 Procesando: [{category}] {pdf_name}")
        print(f"💾 Archivo de salida: {output_filepath}")
        print(f"{'='*60}")

        # --- SOLO PROCESAR EJERCICIOS POR AHORA ---
        if category == "Ejercicios":
            # Crear archivo individual para este PDF
            with open(output_filepath, "w", encoding="utf-8") as output_f:
                output_f.write(f"# Resultados del Preprocesamiento: [{category}] {pdf_name}\n")
                output_f.write(f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                start_time = time.time()
                try:
                    # 2. Extraer texto
                    texto_markdown_raw, metodo = extraer_texto_docling_primero(pdf_path)
                    print(f"  ⏱️ Tiempo de extracción ({metodo}): {time.time() - start_time:.2f} segundos")
                    output_f.write(f"**Método de extracción usado:** {metodo}\n\n")

                    if metodo == "Fallo_Total":
                        output_f.write("```\n" + texto_markdown_raw + "\n```\n") # Escribir el mensaje de error
                        continue # Pasar al siguiente archivo

                    # 3. Quitar imágenes
                    texto_markdown_limpio = quitar_imagenes_base64_markdown(texto_markdown_raw)

                    # --- INICIO: DEBUG TEMPORAL (Mantenido) ---
                    archivos_debug = ["ProblemasTema1.pdf", "ProblemasTema4-1.pdf"]
                    if pdf_path.name in archivos_debug:
                        print(f"\n--- DEBUG: Inicio Markdown Limpio para {pdf_path.name} ({category}) ---")
                        lineas_markdown = texto_markdown_limpio.splitlines()
                        for i, linea in enumerate(lineas_markdown[:50]): # Imprimir primeras 50 líneas
                            print(f"{i+1:02d}: {linea}")
                        print(f"--- DEBUG: Fin Markdown Limpio para {pdf_path.name} ---\n")
                    # --- FIN: DEBUG TEMPORAL ---

                    # 4. Parsear secciones (usando la nueva función específica para ejercicios)
                    parse_start_time = time.time()
                    print("  Parsing específico para EJERCICIOS...")
                    ejercicios_detectados = parsear_ejercicios_v2(texto_markdown_limpio) # LLAMADA A LA NUEVA FUNCIÓN
                    print(f"  ⏱️ Tiempo de parseo: {time.time() - parse_start_time:.2f} segundos")

                    # 5. Escribir secciones estructuradas (formato de ejercicios) en el archivo de salida
                    output_f.write(f"**Ejercicios Detectados:** {len(ejercicios_detectados)}\n\n")
                    if ejercicios_detectados:
                        for i, ejercicio_data in enumerate(ejercicios_detectados):
                            num_ej = ejercicio_data['numero']
                            texto_ej = ejercicio_data['texto']
                            # Usar el número detectado en el encabezado Markdown
                            output_f.write(f"### 🔖 Ejercicio {num_ej}\n\n")
                            output_f.write("```markdown\n")
                            output_f.write(texto_ej + "\n") # Escribir el texto completo del ejercicio
                            output_f.write("```\n\n")
                            # Imprimir prévisualización de los primeros ejercicios en consola
                            if i < 2: # Mostrar solo los dos primeros ejercicios detectados por archivo
                                print(f"  -> Ejercicio {num_ej}: {texto_ej[:200].replace(chr(10), ' ')}...")
                    else:
                        output_f.write("*No se detectaron ejercicios estructurados.*\n\n")
                        print("  ⚠️ No se detectaron ejercicios estructurados.")

                except Exception as e:
                    error_msg = f"  ❌ ERROR Inesperado procesando {pdf_name}: {e}"
                    print(error_msg)
                    traceback.print_exc() # Imprime el stack trace completo en la consola
                    output_f.write(f"**ERROR INESPERADO:**\n```\n{traceback.format_exc()}\n```\n")

                total_time = time.time() - start_time
                print(f"  ✅ Procesamiento de {pdf_name} completado en {total_time:.2f} segundos.")

        else:
            # --- SALTAR OTRAS CATEGORÍAS ---
            print(f"  🚫 Saltando PDF de categoría '{category}' (enfoque actual en 'Ejercicios').")
            # Crear un archivo mínimo indicando que se saltó
            with open(output_filepath, "w", encoding="utf-8") as output_f:
                output_f.write(f"# Archivo: [{category}] {pdf_name}\n\n")
                output_f.write(f"_(Procesamiento saltado: Enfoque actual en 'Ejercicios')_\n\n")
            continue # Pasar al siguiente archivo

    # ... (Mensaje final) ...
    print(f"\n{'='*60}")
    print(f"🎉 Preprocesamiento finalizado.")
    print(f"👉 Revisa los resultados estructurados en: {OUTPUT_DIR}")
    print(f"{'='*60}")
