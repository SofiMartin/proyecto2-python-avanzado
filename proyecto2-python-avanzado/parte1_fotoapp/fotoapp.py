"""
fotoapp.py
==========

App "fotoapp" para editar fotos y prepararlas para redes sociales
(Proyecto 2 - Programación Python Avanzada, TUTP 2026).

Funciones (una por cada punto de la consigna):

1. redimensionar_para_red(ruta, red_social) -> lee la foto (archivo, Drive o URL)
   y la lleva al tamaño de la red social sin deformarla.
2. ecualizar_histograma(imagen)             -> ajusta el contraste con el
   histograma y muestra la foto original y la ecualizada en una misma figura.
3. aplicar_filtro(imagen, filtro)           -> aplica uno de los 9 filtros de
   Pillow y muestra la original y la filtrada lado a lado.
4. boceto(imagen, tecnica)                  -> resalta bordes y contornos
   (umbral, Sobel, Laplaciano o Canny). comparar_bocetos() los muestra juntos.
5. main() / menu()                          -> menú por consola que obliga a
   pasar primero por la función 1. Guarda los resultados en "fotos_mejoradas".

Uso desde la terminal:
    python fotoapp.py
"""

import os
from io import BytesIO

import cv2  # OpenCV: ecualización y bordes
import matplotlib.pyplot as plt  # gráficos y figuras
import numpy as np  # las imágenes se tratan como arrays
import requests  # para descargar imágenes desde una URL
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Excepciones propias
# ---------------------------------------------------------------------------

class ImagenNoCargadaError(Exception):
    """Se quiso editar una imagen sin haberla leído antes con la función 1."""


class RedSocialInvalidaError(ValueError):
    """La palabra clave no es una red social válida."""


class FiltroInvalidoError(ValueError):
    """El filtro pedido no es uno de los 9 disponibles."""


class TecnicaInvalidaError(ValueError):
    """La técnica de boceto pedida no existe."""


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Tamaño recomendado (ancho, alto) en píxeles para cada red social
DIMENSIONES_REDES = {
    "youtube": (1280, 720),     # miniatura de video, 16:9
    "instagram": (1080, 1080),  # publicación cuadrada, 1:1
    "twitter": (1600, 900),     # imagen del timeline, 16:9
    "facebook": (1200, 630),    # imagen compartida, ~1.91:1
}

# Otros nombres aceptados para la misma red
ALIAS_REDES = {"x": "twitter", "ig": "instagram", "fb": "facebook", "yt": "youtube"}

# Los 9 filtros de Pillow, numerados para que el usuario elija por número
FILTROS = {
    1: ("Blur", ImageFilter.BLUR),
    2: ("Contour", ImageFilter.CONTOUR),
    3: ("Detail", ImageFilter.DETAIL),
    4: ("Edge Enhance", ImageFilter.EDGE_ENHANCE),
    5: ("Edge Enhance More", ImageFilter.EDGE_ENHANCE_MORE),
    6: ("Emboss", ImageFilter.EMBOSS),
    7: ("Find Edges", ImageFilter.FIND_EDGES),
    8: ("Sharpen", ImageFilter.SHARPEN),
    9: ("Smooth", ImageFilter.SMOOTH),
}

# Técnicas de boceto con sus parámetros por defecto
TECNICAS_BOCETO = {
    "umbral": {"blur": 5, "bloque": 9, "c": 5},
    "sobel": {"blur": 5, "ksize": 3, "umbral": 60},
    "laplaciano": {"blur": 5, "ksize": 3, "umbral": 20},
    "canny": {"blur": 5, "umbral_bajo": 50, "umbral_alto": 150},
}

# Carpeta donde el menú guarda todos los resultados
CARPETA_SALIDA = "fotos_mejoradas"


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def _validar_imagen(imagen):
    """Lanza ImagenNoCargadaError si `imagen` no es una imagen de Pillow ya leída."""
    if not isinstance(imagen, Image.Image):
        raise ImagenNoCargadaError(
            "No hay ninguna imagen cargada. Primero leé la foto con la opción 1 "
            "(redimensionar_para_red)."
        )


def _crear_carpeta_de(ruta_archivo):
    """Crea la carpeta donde va a quedar el archivo (si no existe)."""
    carpeta = os.path.dirname(ruta_archivo)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)


def _guardar_figura(fig, ruta_figura, mostrar):
    """Guarda la figura (si hay ruta) y la muestra o la cierra."""
    if ruta_figura:
        _crear_carpeta_de(ruta_figura)
        fig.savefig(ruta_figura, dpi=100, bbox_inches="tight")
    if mostrar:
        plt.show()
    else:
        plt.close(fig)


def _impar(valor):
    """OpenCV pide tamaños de kernel impares: 4 -> 5, 5 -> 5."""
    valor = int(valor)
    return valor if valor % 2 == 1 else valor + 1


def leer_imagen(ruta):
    """Lee una imagen desde un archivo, Google Drive (montado) o una URL.

    Devuelve la imagen de Pillow en modo RGB.
    """
    if not isinstance(ruta, str) or not ruta.strip():
        raise ValueError("La ruta de la imagen debe ser un texto no vacío.")
    ruta = ruta.strip()

    if ruta.lower().startswith(("http://", "https://")):
        # Es una URL: descargamos los bytes y los abrimos como archivo
        try:
            respuesta = requests.get(ruta, headers={"User-Agent": "fotoapp/1.0"}, timeout=20)
            respuesta.raise_for_status()
        except requests.RequestException as error:
            raise ValueError(f"No se pudo descargar la imagen desde la URL: {error}") from error
        origen = BytesIO(respuesta.content)
    else:
        # Es un archivo local o de Drive
        if not os.path.isfile(ruta):
            mensaje = f"No existe el archivo '{ruta}'."
            if ruta.startswith("/content/drive"):
                mensaje += " ¿Montaste Google Drive? (drive.mount('/content/drive'))"
            raise FileNotFoundError(mensaje)
        origen = ruta

    try:
        imagen = Image.open(origen)
    except UnidentifiedImageError as error:
        raise ValueError(f"'{ruta}' no es un archivo de imagen válido.") from error
    imagen = ImageOps.exif_transpose(imagen)  # respeta la rotación de la cámara
    return imagen.convert("RGB")


def normalizar_red(red_social):
    """Convierte la palabra clave a la clave interna ('Instagram ' -> 'instagram')."""
    if not isinstance(red_social, str):
        raise RedSocialInvalidaError("La red social debe ser un texto.")
    clave = red_social.strip().lower()
    clave = ALIAS_REDES.get(clave, clave)
    if clave not in DIMENSIONES_REDES:
        opciones = ", ".join(r.capitalize() for r in DIMENSIONES_REDES)
        raise RedSocialInvalidaError(f"Red social '{red_social}' no soportada. Opciones: {opciones}.")
    return clave


# ---------------------------------------------------------------------------
# 1. Leer y redimensionar para una red social
# ---------------------------------------------------------------------------

def redimensionar_para_red(ruta, red_social, modo="recortar", ruta_salida=None, mostrar=False):
    """Lee una foto y la redimensiona al tamaño recomendado de una red social.

    Parámetros
    ----------
    ruta : str
        Ruta del archivo, ruta de Google Drive o URL de la imagen.
    red_social : str
        'Youtube', 'Instagram', 'Twitter' o 'Facebook' (sin distinguir mayúsculas).
    modo : str
        Cómo se mantiene la proporción para no distorsionar la foto:
        - 'recortar' (por defecto): agranda la foto hasta cubrir el tamaño
          final y recorta lo que sobra, centrado.
        - 'ajustar': achica la foto hasta que entre completa y rellena con
          bandas blancas lo que sobra.
    ruta_salida : str, opcional
        Si se indica, guarda la imagen redimensionada en ese archivo.
    mostrar : bool
        Si es True, muestra la original y la redimensionada lado a lado.

    Devuelve la imagen (PIL.Image.Image) con el tamaño exacto de la red social.
    """
    clave = normalizar_red(red_social)
    ancho, alto = DIMENSIONES_REDES[clave]      # tamaño final (ancho, alto)
    original = leer_imagen(ruta)

    # Factores de escala para llegar al ancho y al alto finales
    escala_ancho = ancho / original.width
    escala_alto = alto / original.height

    modo = modo.strip().lower()
    if modo == "recortar":
        # Usamos la escala MAYOR para que la foto cubra todo el tamaño final
        escala = max(escala_ancho, escala_alto)
        nuevo_ancho = max(ancho, round(original.width * escala))
        nuevo_alto = max(alto, round(original.height * escala))
        grande = original.resize((nuevo_ancho, nuevo_alto), Image.LANCZOS)
        # Recortamos el centro, con el tamaño exacto
        izquierda = (nuevo_ancho - ancho) // 2
        arriba = (nuevo_alto - alto) // 2
        resultado = grande.crop((izquierda, arriba, izquierda + ancho, arriba + alto))
    elif modo == "ajustar":
        # Usamos la escala MENOR para que la foto entre completa
        escala = min(escala_ancho, escala_alto)
        chica = original.resize((round(original.width * escala), round(original.height * escala)),
                                Image.LANCZOS)
        # Pegamos la foto centrada sobre un fondo blanco del tamaño final
        resultado = Image.new("RGB", (ancho, alto), "white")
        resultado.paste(chica, ((ancho - chica.width) // 2, (alto - chica.height) // 2))
    else:
        raise ValueError("modo debe ser 'recortar' o 'ajustar'.")

    if ruta_salida:
        _crear_carpeta_de(ruta_salida)
        resultado.save(ruta_salida)

    if mostrar:
        _, ejes = plt.subplots(1, 2, figsize=(12, 5))
        ejes[0].imshow(original)
        ejes[0].set_title(f"Original {original.width}x{original.height}")
        ejes[1].imshow(resultado)
        ejes[1].set_title(f"{clave.capitalize()} {ancho}x{alto} ({modo})")
        for eje in ejes:
            eje.axis("off")
        plt.tight_layout()
        plt.show()

    return resultado


# ---------------------------------------------------------------------------
# 2. Ecualización de histograma
# ---------------------------------------------------------------------------

def ecualizar_imagen(imagen):
    """Devuelve una copia de la imagen con el contraste ecualizado.

    cv2.equalizeHist solo funciona en grises, así que en una foto color se
    ecualiza solo el brillo (canal Y de YCrCb) y se dejan los colores como están.
    """
    _validar_imagen(imagen)
    arreglo = np.array(imagen.convert("RGB"))
    ycrcb = cv2.cvtColor(arreglo, cv2.COLOR_RGB2YCrCb)    # separa brillo y color
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])     # ecualiza solo el brillo
    return Image.fromarray(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB))


def ecualizar_histograma(imagen, ruta_figura=None, mostrar=True, titulo=""):
    """Ajusta el contraste de la foto con su histograma y grafica el resultado.

    Muestra en una misma figura la foto original y la ecualizada, cada una con
    su histograma. Si se indica `ruta_figura`, guarda la figura.

    Devuelve la imagen ecualizada.
    """
    _validar_imagen(imagen)
    ecualizada = ecualizar_imagen(imagen)

    fig, ejes = plt.subplots(2, 2, figsize=(13, 9))
    # Arriba: las dos fotos
    ejes[0, 0].imshow(imagen)
    ejes[0, 0].set_title("Original")
    ejes[0, 1].imshow(ecualizada)
    ejes[0, 1].set_title("Ecualizada")
    ejes[0, 0].axis("off")
    ejes[0, 1].axis("off")
    # Abajo: el histograma de cada una (en escala de grises)
    for eje, foto, nombre in [(ejes[1, 0], imagen, "Histograma original"),
                              (ejes[1, 1], ecualizada, "Histograma ecualizado")]:
        gris = np.array(foto.convert("L"))
        eje.hist(gris.ravel(), bins=256, range=(0, 256), color="gray")
        eje.set_title(nombre)
        eje.set_xlabel("Intensidad (0 = negro, 255 = blanco)")
        eje.set_ylabel("Cantidad de píxeles")
    fig.suptitle(f"Ecualización de histograma {('- ' + titulo) if titulo else ''}", fontsize=14)
    fig.tight_layout()
    _guardar_figura(fig, ruta_figura, mostrar)
    return ecualizada


# ---------------------------------------------------------------------------
# 3. Filtros de Pillow
# ---------------------------------------------------------------------------

def listar_filtros():
    """Devuelve un texto con los 9 filtros numerados (para el menú)."""
    return "\n".join(f"  {numero}. {nombre}" for numero, (nombre, _) in FILTROS.items())


def obtener_filtro(filtro):
    """Acepta el número (1-9) o el nombre del filtro y devuelve (nombre, filtro_pillow)."""
    if isinstance(filtro, str) and filtro.strip().isdigit():
        filtro = int(filtro.strip())            # "8" -> 8
    if isinstance(filtro, int):
        if filtro in FILTROS:
            return FILTROS[filtro]
    elif isinstance(filtro, str):
        buscado = filtro.strip().lower().replace("_", " ")
        for nombre, filtro_pillow in FILTROS.values():
            if nombre.lower() == buscado:
                return nombre, filtro_pillow
    raise FiltroInvalidoError(f"Filtro '{filtro}' inválido. Elegí un número o nombre de:\n{listar_filtros()}")


def aplicar_filtro(imagen, filtro, ruta_figura=None, mostrar=True):
    """Aplica uno de los 9 filtros de Pillow a la imagen.

    `filtro` puede ser el nombre ('Emboss') o el número (1-9). Muestra la
    original y la filtrada lado a lado (la segunda con el nombre del filtro
    como título) y guarda la figura si se indica `ruta_figura`.

    Devuelve la imagen filtrada.
    """
    _validar_imagen(imagen)
    nombre, filtro_pillow = obtener_filtro(filtro)
    filtrada = imagen.filter(filtro_pillow)     # aplica el filtro de Pillow

    fig, ejes = plt.subplots(1, 2, figsize=(12, 6))
    ejes[0].imshow(imagen)
    ejes[0].set_title("Original")
    ejes[1].imshow(filtrada)
    ejes[1].set_title(nombre)
    for eje in ejes:
        eje.axis("off")
    fig.tight_layout()
    _guardar_figura(fig, ruta_figura, mostrar)
    return filtrada


# ---------------------------------------------------------------------------
# 4. Boceto de líneas (bordes y contornos)
# ---------------------------------------------------------------------------

def boceto(imagen, tecnica="canny", grosor=1, **parametros):
    """Devuelve un boceto de líneas negras sobre fondo blanco.

    Técnicas (parámetros por defecto en TECNICAS_BOCETO):
    - 'umbral'    : umbralización adaptativa (compara cada píxel con el promedio
                    de sus vecinos: `bloque` y `c`).
    - 'sobel'     : gradiente de Sobel; se quedan los bordes mayores a `umbral`.
    - 'laplaciano': Laplaciano; se quedan los bordes mayores a `umbral`.
    - 'canny'     : detector de Canny (`umbral_bajo` y `umbral_alto`).

    En todas, `blur` es el desenfoque previo que saca el ruido y `grosor` > 1
    engrosa las líneas.
    """
    _validar_imagen(imagen)
    tecnica = tecnica.strip().lower()
    if tecnica not in TECNICAS_BOCETO:
        raise TecnicaInvalidaError(f"Técnica '{tecnica}' inválida. Opciones: {', '.join(TECNICAS_BOCETO)}.")
    p = {**TECNICAS_BOCETO[tecnica], **parametros}   # defaults + lo que pase el usuario

    gris = np.array(imagen.convert("L"))                               # pasamos a grises
    k = _impar(p["blur"])
    suave = cv2.GaussianBlur(gris, (k, k), 0)                          # sacamos ruido

    if tecnica == "umbral":
        resultado = cv2.adaptiveThreshold(suave, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, _impar(p["bloque"]), p["c"])
        bordes = resultado == 0                                        # lo negro son los bordes
    elif tecnica == "sobel":
        gx = cv2.Sobel(suave, cv2.CV_64F, 1, 0, ksize=_impar(p["ksize"]))   # derivada en x
        gy = cv2.Sobel(suave, cv2.CV_64F, 0, 1, ksize=_impar(p["ksize"]))   # derivada en y
        magnitud = np.sqrt(gx ** 2 + gy ** 2)                          # fuerza del borde
        magnitud = magnitud / max(magnitud.max(), 1) * 255             # llevamos a 0..255
        bordes = magnitud > p["umbral"]
    elif tecnica == "laplaciano":
        lap = np.abs(cv2.Laplacian(suave, cv2.CV_64F, ksize=_impar(p["ksize"])))
        lap = lap / max(lap.max(), 1) * 255                            # llevamos a 0..255
        bordes = lap > p["umbral"]
    else:  # canny
        bordes = cv2.Canny(suave, p["umbral_bajo"], p["umbral_alto"]) > 0

    if grosor > 1:
        # Engrosamos las líneas con dilatación
        bordes = cv2.dilate(bordes.astype(np.uint8), np.ones((int(grosor), int(grosor)), np.uint8)) > 0

    salida = np.where(bordes, 0, 255).astype(np.uint8)   # líneas negras, fondo blanco
    return Image.fromarray(salida)


def comparar_bocetos(imagen, configuraciones, ruta_figura=None, mostrar=True, columnas=3):
    """Muestra lado a lado la imagen original y varios bocetos.

    `configuraciones` es una lista de diccionarios, por ejemplo:
        [{"tecnica": "canny", "umbral_bajo": 50, "umbral_alto": 150},
         {"tecnica": "sobel", "umbral": 60}]

    Devuelve la lista de bocetos generados.
    """
    _validar_imagen(imagen)
    total = len(configuraciones) + 1                    # +1 por la original
    filas = int(np.ceil(total / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(5 * columnas, 5 * filas))
    ejes = np.atleast_1d(ejes).ravel()

    ejes[0].imshow(imagen)
    ejes[0].set_title("Original")

    bocetos = []
    for eje, config in zip(ejes[1:], configuraciones):
        config = dict(config)
        tecnica = config.pop("tecnica")                 # el resto son parámetros
        resultado = boceto(imagen, tecnica, **config)
        bocetos.append(resultado)
        eje.imshow(resultado, cmap="gray", vmin=0, vmax=255)
        detalle = ", ".join(f"{k}={v}" for k, v in config.items())
        eje.set_title(f"{tecnica}\n{detalle}" if detalle else tecnica, fontsize=10)

    for eje in ejes:
        eje.axis("off")
    fig.tight_layout()
    _guardar_figura(fig, ruta_figura, mostrar)
    return bocetos


# ---------------------------------------------------------------------------
# 5. Menú de opciones
# ---------------------------------------------------------------------------

OPCIONES_MENU = """
================ FOTOAPP ================
Imagen actual: {estado}
  1. Leer imagen y redimensionar para una red social
  2. Ajustar contraste (ecualizar histograma)
  3. Aplicar un filtro
  4. Generar boceto (bordes y contornos)
  5. Guardar imagen actual
  0. Salir
========================================="""


def _preguntar_si(entrada, pregunta):
    """Hace una pregunta s/n y devuelve True si la respuesta es sí."""
    return entrada(pregunta + " (s/n): ").strip().lower() in ("s", "si", "sí", "y", "yes")


def menu(entrada=input, carpeta_salida=CARPETA_SALIDA, mostrar=True):
    """Menú interactivo de fotoapp.

    Para las opciones 2 a 5 hace falta haber leído una imagen con la opción 1.
    Si el usuario se la salta, las funciones lanzan ImagenNoCargadaError y el
    menú le explica qué hacer.

    `entrada` es la función que lee lo que escribe el usuario (por defecto
    `input`). El notebook la reemplaza por respuestas simuladas para probar
    el menú de forma automática.
    """
    imagen_actual = None                        # todavía no se cargó ninguna imagen
    descripcion = "ninguna (usá la opción 1)"
    contador = 0                                # para numerar los archivos guardados

    def ruta_nueva(nombre):
        nonlocal contador
        contador += 1
        return os.path.join(carpeta_salida, f"menu_{contador:02d}_{nombre}.png")

    while True:
        print(OPCIONES_MENU.format(estado=descripcion))
        try:
            opcion = entrada("Elegí una opción: ").strip()

            if opcion == "0":
                print("¡Hasta luego!")
                break

            elif opcion == "1":
                # Leer la foto y redimensionarla para la red social elegida
                ruta = entrada("Ruta (archivo, Drive) o URL de la imagen: ")
                red = entrada("Red social (Youtube / Instagram / Twitter / Facebook): ")
                modo = entrada("Modo [r]ecortar o [a]justar con bandas (Enter = recortar): ")
                modo = "ajustar" if modo.strip().lower().startswith("a") else "recortar"
                clave = normalizar_red(red)
                nombre_base = os.path.splitext(os.path.basename(ruta.strip()))[0] or "foto"
                ruta_salida = os.path.join(carpeta_salida, f"{clave}_{nombre_base}.jpg")
                imagen_actual = redimensionar_para_red(ruta, red, modo=modo,
                                                       ruta_salida=ruta_salida, mostrar=mostrar)
                descripcion = f"{os.path.basename(ruta.strip())} -> {clave.capitalize()} {imagen_actual.size}"
                print(f"Imagen lista para {clave.capitalize()}: {imagen_actual.size[0]}x{imagen_actual.size[1]} px")
                print(f"Imagen guardada en {ruta_salida}")

            elif opcion == "2":
                # Ecualizar el histograma
                _validar_imagen(imagen_actual)
                ruta_figura = ruta_nueva("ecualizacion")
                resultado = ecualizar_histograma(imagen_actual, ruta_figura, mostrar=mostrar)
                print(f"Figura guardada en {ruta_figura}")
                if _preguntar_si(entrada, "¿Usar la imagen ecualizada como imagen actual?"):
                    imagen_actual = resultado
                    descripcion += " + ecualizada"

            elif opcion == "3":
                # Aplicar un filtro
                _validar_imagen(imagen_actual)
                print("Filtros disponibles:\n" + listar_filtros())
                nombre, _ = obtener_filtro(entrada("Número de filtro (1-9): "))
                ruta_figura = ruta_nueva("filtro_" + nombre.lower().replace(" ", "_"))
                resultado = aplicar_filtro(imagen_actual, nombre, ruta_figura, mostrar=mostrar)
                print(f"Figura guardada en {ruta_figura}")
                if _preguntar_si(entrada, f"¿Usar la imagen con {nombre} como imagen actual?"):
                    imagen_actual = resultado
                    descripcion += f" + {nombre}"

            elif opcion == "4":
                # Generar un boceto
                _validar_imagen(imagen_actual)
                tecnicas = list(TECNICAS_BOCETO)
                print("Técnicas:\n" + "\n".join(f"  {i}. {t}" for i, t in enumerate(tecnicas, 1)))
                eleccion = entrada("Número de técnica (Enter = canny): ").strip() or str(len(tecnicas))
                if not eleccion.isdigit() or not 1 <= int(eleccion) <= len(tecnicas):
                    raise TecnicaInvalidaError(f"Opción de técnica '{eleccion}' inválida.")
                tecnica = tecnicas[int(eleccion) - 1]
                ruta_figura = ruta_nueva("boceto_" + tecnica)
                resultado = comparar_bocetos(imagen_actual, [{"tecnica": tecnica}], ruta_figura,
                                             mostrar=mostrar, columnas=2)[0]
                print(f"Figura guardada en {ruta_figura}")
                if _preguntar_si(entrada, "¿Usar el boceto como imagen actual?"):
                    imagen_actual = resultado.convert("RGB")
                    descripcion += f" + boceto {tecnica}"

            elif opcion == "5":
                # Guardar la imagen actual
                _validar_imagen(imagen_actual)
                nombre = entrada("Nombre del archivo (Enter = automático): ").strip()
                ruta = os.path.join(carpeta_salida, nombre) if nombre else ruta_nueva("imagen")
                if not os.path.splitext(ruta)[1]:
                    ruta += ".png"
                _crear_carpeta_de(ruta)
                imagen_actual.save(ruta)
                print(f"Imagen guardada en {ruta}")

            else:
                print("Opción inválida. Escribí un número del 0 al 5.")

        except ImagenNoCargadaError as error:
            print(f"[!] {error}")
        except (ValueError, FileNotFoundError, OSError) as error:
            print(f"[!] Error: {error}")
        except (KeyboardInterrupt, EOFError):
            print("\nSaliendo de fotoapp.")
            break


def main():
    """Punto de entrada de la app: inicia el menú por consola."""
    menu()


if __name__ == "__main__":
    main()
