"""Junta las secciones del informe en un solo HTML listo para exportar a PDF.

Responsable: Marinés

Uso, desde la raíz del repo:
    pip install markdown
    python informe/armar_informe.py

Genera informe/Informe_Proyecto3.html con todas las imágenes incrustadas.
Para el PDF: abrir el HTML en el navegador → Imprimir → Guardar como PDF
(tamaño A4, márgenes "predeterminados", activar "gráficos de fondo").

Qué hace además de convertir el Markdown:
  - Portada e índice automáticos.
  - <!-- incluir: ruta.md --> inserta otro archivo (p. ej. la tabla comparativa
    que genera run_benchmarks.py), así el informe siempre tiene los números actuales.
  - Quita las notas internas (citas que empiezan con ⚠️).
  - Agrega las capturas de evidencias/ como anexo.
  - Avisa en la terminal si falta una sección o si quedan ⟨…⟩ por llenar.
"""

import base64
import html
import mimetypes
import re
import sys
from pathlib import Path

try:
    import markdown
except ImportError:
    sys.exit("Falta la librería markdown: pip install markdown")

CARPETA = Path(__file__).resolve().parent
RAIZ = CARPETA.parent
SALIDA = CARPETA / "Informe_Proyecto3.html"

TITULO = "Proyecto 3: Mejorando algoritmos con CUDA"
SUBTITULO = "Histograma de temperaturas con MergeSort: versión secuencial vs CPU multinúcleo + GPU"
CURSO = "CC3069 Computación Paralela y Distribuida · Universidad del Valle de Guatemala"
INTEGRANTES = [("Marinés García", "23391"), ("Camila Richter", "23183"), ("Carlos Alburez", "231311")]
FECHA = "18 de noviembre de 2026"

# Orden del informe: (archivo, nombre de la sección para los avisos)
SECCIONES = [
    ("01_analisis_secuencial.md", "Paso 1 · Análisis secuencial (Marinés)"),
    ("02_tiempos_secuenciales.md", "Paso 2 · Tiempos secuenciales (Marinés)"),
    ("03_oportunidades.md", "Paso 3 · Oportunidades de paralelización (Camila)"),
    ("04_diseno_paralelo.md", "Paso 4 · Diseño del algoritmo paralelo (Camila)"),
    ("05_implementacion_cuda.md", "Paso 5 · Implementación con CUDA Python (Carlos)"),
    ("06_resultados.md", "Paso 6 · Comparación y speedup (Marinés)"),
    ("06_discusion.md", "Paso 6 · Discusión técnica (los 3)"),
]

CSS = """
@page { size: A4; margin: 18mm 16mm; }
:root { --texto: #1d2733; --suave: #5b6673; --linea: #d7dde4; --acento: #1f6fb2; --fondo-tabla: #f3f6f9; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: var(--texto);
       background: #fff; font-size: 10.5pt; line-height: 1.5; max-width: 820px; margin: 0 auto; padding: 24px; }
h1 { font-size: 19pt; color: var(--acento); border-bottom: 2px solid var(--acento); padding-bottom: 4px;
     margin-top: 0; page-break-before: always; break-before: page; }
h2 { font-size: 14pt; margin-top: 1.6em; }
h3 { font-size: 11.5pt; margin-top: 1.3em; }
h1, h2, h3 { page-break-after: avoid; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
table { border-collapse: collapse; width: 100%; margin: 0.8em 0 1.2em; font-size: 9pt;
        page-break-inside: avoid; break-inside: avoid; }
th, td { border: 1px solid var(--linea); padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: var(--fondo-tabla); }
tr:nth-child(even) td { background: #fafbfc; }
img { max-width: 100%; display: block; margin: 0.8em auto 0.2em; page-break-inside: avoid; break-inside: avoid; }
p > em:only-child { display: block; text-align: center; color: var(--suave); font-size: 9pt; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.8pt; background: #f1f3f5; padding: 1px 4px; border-radius: 3px; }
pre { background: #f6f8fa; border: 1px solid var(--linea); padding: 8px 10px; overflow-x: auto;
      font-size: 8.5pt; page-break-inside: avoid; break-inside: avoid; }
pre code { background: none; padding: 0; }
blockquote { margin: 0.6em 0; padding: 4px 12px; border-left: 3px solid var(--linea); color: var(--suave); }
.portada { text-align: center; padding-top: 140px; page-break-after: always; break-after: page; }
.portada h1 { border: none; font-size: 24pt; page-break-before: avoid; break-before: avoid; }
.portada .sub { font-size: 13pt; color: var(--suave); margin-bottom: 70px; }
.portada table { width: 60%; margin: 0 auto 50px; font-size: 11pt; }
.indice { page-break-after: always; break-after: page; }
.indice h1 { page-break-before: avoid; break-before: avoid; }
.indice ul { list-style: none; padding-left: 0; }
.indice ul ul { padding-left: 18px; }
.indice a { color: var(--texto); text-decoration: none; }
.pendiente { border: 2px dashed #c0392b; color: #c0392b; padding: 12px; margin: 1em 0; }
.evidencia { page-break-inside: avoid; break-inside: avoid; margin-bottom: 1.4em; }
@media print { body { padding: 0; max-width: none; } a { color: inherit; } }
"""

MATHJAX = """<script>window.MathJax = { tex: { inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]'], ['$$', '$$']] } };</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"></script>"""

avisos = []


def incluir_archivos(texto, base):
    """Reemplaza <!-- incluir: ruta --> por el contenido de ese archivo."""
    def reemplazo(m):
        ruta = (base / m.group(1).strip()).resolve()
        if ruta.exists():
            return ruta.read_text(encoding="utf-8")
        avisos.append(f"No existe {ruta.relative_to(RAIZ)} (se incluye desde {base.name}): falta generarlo")
        return f'\n<div class="pendiente">Pendiente: {html.escape(str(ruta.relative_to(RAIZ)))}</div>\n'
    return re.sub(r"<!--\s*incluir:\s*(.+?)\s*-->", reemplazo, texto)


def quitar_notas_internas(texto):
    """Quita los bloques de cita (>) que contienen ⚠️: son notas para el equipo."""
    salida, bloque = [], []

    def cerrar():
        if bloque and not any("⚠️" in linea for linea in bloque):
            salida.extend(bloque)
        bloque.clear()

    for linea in texto.splitlines():
        if linea.lstrip().startswith(">"):
            bloque.append(linea)
        else:
            cerrar()
            salida.append(linea)
    cerrar()
    return "\n".join(salida)


def proteger_formulas(texto):
    """Markdown se come las barras de \\[ \\]; se guardan aparte y se restauran después."""
    guardadas = []

    def guardar(m):
        guardadas.append(m.group(0))
        return f"FORMULA{len(guardadas) - 1}FIN"

    texto = re.sub(r"\\\[.+?\\\]|\\\(.+?\\\)|\$\$.+?\$\$", guardar, texto, flags=re.S)
    return texto, guardadas


def restaurar_formulas(texto, guardadas):
    return re.sub(r"FORMULA(\d+)FIN", lambda m: html.escape(guardadas[int(m.group(1))], quote=False), texto)


def incrustar_imagenes(html_texto, base):
    """Convierte <img src="figuras/x.png"> en data: URI para que el HTML sea autocontenido."""
    def reemplazo(m):
        src = m.group(2)
        if src.startswith(("http:", "https:", "data:")):
            return m.group(0)
        ruta = (base / src).resolve()
        if not ruta.exists():
            avisos.append(f"Falta la imagen {src} (en {base.name})")
            return m.group(0)
        tipo = mimetypes.guess_type(ruta.name)[0] or "image/png"
        datos = base64.b64encode(ruta.read_bytes()).decode()
        return f'{m.group(1)}data:{tipo};base64,{datos}{m.group(3)}'
    return re.sub(r'(<img[^>]*src=")([^"]+)(")', reemplazo, html_texto)


def convertir(texto_md):
    texto_md, formulas = proteger_formulas(texto_md)
    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "toc"],
                           extension_configs={"toc": {"slugify": slug}})
    cuerpo = md.convert(texto_md)
    return restaurar_formulas(cuerpo, formulas)


def slug(valor, separador="-"):
    valor = re.sub(r"[^\w\s-]", "", valor.lower(), flags=re.U).strip()
    return re.sub(r"[\s_-]+", separador, valor)


def portada():
    filas = "".join(f"<tr><td>{html.escape(n)}</td><td>{c}</td></tr>" for n, c in INTEGRANTES)
    return f"""<section class="portada">
<p>{html.escape(CURSO)}</p>
<h1>{html.escape(TITULO)}</h1>
<p class="sub">{html.escape(SUBTITULO)}</p>
<table><tr><th>Integrante</th><th>Carné</th></tr>{filas}</table>
<p>{html.escape(FECHA)}</p>
</section>"""


def indice(cuerpo):
    """Índice con los h1 y h2 del documento."""
    items, abierto = [], False
    for nivel, ident, titulo in re.findall(r'<h([12]) id="([^"]+)">(.+?)</h\1>', cuerpo):
        texto = re.sub(r"<[^>]+>", "", titulo)
        if nivel == "1":
            if abierto:
                items.append("</ul></li>")
            items.append(f'<li><a href="#{ident}"><strong>{texto}</strong></a><ul>')
            abierto = True
        else:
            items.append(f'<li><a href="#{ident}">{texto}</a></li>')
    if abierto:
        items.append("</ul></li>")
    return f'<section class="indice"><h1>Índice</h1><ul>{"".join(items)}</ul></section>'


def anexo_evidencias():
    carpeta = RAIZ / "evidencias"
    imagenes = sorted(p for p in carpeta.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    logs = sorted(p for p in carpeta.glob("*") if p.suffix.lower() in (".txt", ".log", ".csv"))
    partes = ["# Anexo: evidencia de ejecución\n",
              "Capturas y registros de los recursos del sistema durante las ejecuciones "
              "(carpeta `evidencias/` del repositorio).\n"]
    if not imagenes and not logs:
        avisos.append("La carpeta evidencias/ está vacía: faltan las capturas de nvidia-smi y CPU")
    for img in imagenes:
        nombre = img.stem.replace("_", " ").replace("-", " ")
        partes.append(f"![{nombre}](../evidencias/{img.name})\n\n*{nombre.capitalize()}*\n")
    if logs:
        partes.append("\nRegistros incluidos en el repositorio: " + ", ".join(f"`evidencias/{p.name}`" for p in logs) + ".\n")
    if not any("nvidia" in p.name.lower() or "gpu" in p.name.lower() for p in imagenes + logs):
        avisos.append("No hay ninguna evidencia de la GPU (nombre con 'nvidia' o 'gpu') en evidencias/")
    return "\n".join(partes)


def main():
    textos = []
    for archivo, nombre in SECCIONES:
        ruta = CARPETA / archivo
        if not ruta.exists():
            avisos.append(f"Falta la sección {nombre}: informe/{archivo}")
            paso, resto = (parte.strip() for parte in nombre.split("·", 1))
            titulo = f"{paso}. {resto.split('(')[0].strip()}"
            textos.append(f'# {titulo}\n\n<div class="pendiente">Sección pendiente: informe/{archivo}</div>\n')
            continue
        texto = incluir_archivos(ruta.read_text(encoding="utf-8"), CARPETA)
        texto = quitar_notas_internas(texto)
        pendientes = texto.count("⟨")
        if pendientes:
            avisos.append(f"{archivo}: quedan {pendientes} valores ⟨…⟩ por llenar")
        textos.append(texto)
    textos.append(anexo_evidencias())

    cuerpo = incrustar_imagenes(convertir("\n\n".join(textos)), CARPETA)
    documento = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Informe Proyecto 3 CUDA</title>
<style>{CSS}</style>
{MATHJAX}
</head><body>
{portada()}
{indice(cuerpo)}
{cuerpo}
</body></html>"""
    SALIDA.write_text(documento, encoding="utf-8")

    print(f"Informe generado: {SALIDA.relative_to(RAIZ)} ({SALIDA.stat().st_size / 1e6:.1f} MB)")
    if avisos:
        print(f"\n⚠️  {len(avisos)} pendiente(s) antes de entregar:")
        for aviso in avisos:
            print(f"   - {aviso}")
    else:
        print("Sin pendientes: abrir en el navegador → Imprimir → Guardar como PDF.")


if __name__ == "__main__":
    main()
