"""
fotoapp.py
==========

Módulo de la App "fotoapp" para editar fotos y prepararlas para compartir en
redes sociales (Proyecto 2 - Programación Python Avanzada, TUTP 2026).

Funciones principales (una por cada punto de la consigna):

1. redimensionar_para_red(ruta, red_social)  -> lee la foto (archivo local,
   Google Drive o URL) y la lleva al tamaño recomendado de la red social
   sin deformarla.
2. ecualizar_histograma(imagen)              -> ajusta el contraste usando el
   histograma y muestra/guarda la figura "original vs ecualizada".
3. aplicar_filtro(imagen, filtro)            -> aplica uno de los 9 filtros de
   Pillow y muestra/guarda la figura "original vs filtrada".
4. boceto(imagen, tecnica, ...)              -> resalta bordes y contornos para
   obtener un boceto de líneas (umbral adaptativo, Sobel, Laplaciano, Canny o
   lápiz). comparar_bocetos() las muestra lado a lado.
5. menu() / main()                           -> menú de opciones por consola que
   obliga a pasar primero por la función 1. Los resultados se guardan en la
   carpeta "fotos_mejoradas".

Uso rápido desde la terminal:
    python fotoapp.py
"""

import os  # manejo de rutas y carpetas
from io import BytesIO  # permite tratar bytes descargados como si fueran un archivo

import cv2  # OpenCV: bordes y contornos (Sobel, Laplaciano, Canny)
import matplotlib.pyplot as plt  # para graficar y comparar imágenes
import numpy as np  # arreglos numéricos (las imágenes son matrices)
import requests  # descarga de imágenes desde una URL
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Excepciones propias
# ---------------------------------------------------------------------------


class FotoAppError(Exception):
    """Error base de fotoapp (todas las excepciones propias heredan de esta)."""


class ImagenNoCargadaError(FotoAppError):
    """Se intentó editar una imagen sin haberla leído antes con la función 1."""


class RedSocialInvalidaError(FotoAppError, ValueError):
    """La palabra clave no corresponde a una red social soportada."""


class FiltroInvalidoError(FotoAppError, ValueError):
    """El filtro pedido no es uno de los 9 filtros disponibles."""


class TecnicaInvalidaError(FotoAppError, ValueError):
    """La técnica de boceto pedida no existe."""


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Tamaños recomendados (ancho, alto) en píxeles para una publicación de imagen:
# - YouTube:   miniatura de video, 1280x720 (16:9).
# - Instagram: publicación cuadrada del feed, 1080x1080 (1:1).
# - Twitter/X: imagen en el timeline, 1600x900 (16:9).
# - Facebook:  imagen compartida / de enlace, 1200x630 (~1.91:1).
DIMENSIONES_REDES = {
    "youtube": (1280, 720),
    "instagram": (1080, 1080),
    "twitter": (1600, 900),
    "facebook": (1200, 630),
}

# Otros nombres que aceptamos para la misma red.
ALIAS_REDES = {"x": "twitter", "ig": "instagram", "fb": "facebook", "yt": "youtube"}

# Los 9 filtros de Pillow pedidos, numerados para que el usuario elija por número.
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

# Técnicas de boceto y sus parámetros por defecto.
TECNICAS_BOCETO = {
    "umbral": {"blur": 5, "bloque": 9, "c": 5},
    "sobel": {"blur": 5, "ksize": 3, "umbral": 60},
    "laplaciano": {"blur": 5, "ksize": 3, "umbral": 20},
    "canny": {"blur": 5, "umbral_bajo": 50, "umbral_alto": 150},
    "lapiz": {"blur": 21},
}

CARPETA_SALIDA = "fotos_mejoradas"  # carpeta donde se guardan todos los resultados

_HEADERS_HTTP = {"User-Agent": "fotoapp/1.0 (proyecto educativo Python Avanzado)"}


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------


def _validar_imagen(imagen):
    """Verifica que `imagen` sea una imagen de Pillow ya cargada.

    Lanza ImagenNoCargadaError si no lo es: así obligamos a pasar primero por
    la función 1 (redimensionar_para_red), que es la que lee la foto.
    """
    if imagen is None:
        raise ImagenNoCargadaError(
            "No hay ninguna imagen cargada. Primero leé la foto con la opción 1 "
            "(redimensionar_para_red)."
        )
    if not isinstance(imagen, Image.Image):
        raise ImagenNoCargadaError(
            f"Se esperaba una imagen de Pillow y se recibió {type(imagen).__name__}. "
            "Usá primero redimensionar_para_red() para leer la foto."
        )


def _crear_carpeta_de(ruta_archivo):
    # Obtiene la carpeta del archivo y la crea si no existe
    carpeta = os.path.dirname(ruta_archivo)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)


def _guardar_figura(fig, ruta_figura, mostrar):
    """Guarda la figura (si se pidió) y la muestra o la cierra."""
    if ruta_figura:  # guarda la figura en disco solo si se pasó una ruta
        _crear_carpeta_de(ruta_figura)
        fig.savefig(ruta_figura, dpi=100, bbox_inches="tight")
    if mostrar:
        plt.show()  # la muestra en pantalla
    else:
        plt.close(fig)  # la cierra para liberar memoria


def leer_imagen(ruta):
    """Lee una imagen desde un archivo local, Google Drive (montado) o una URL.

    Devuelve la imagen en modo RGB y con la orientación EXIF corregida.
    """
    # Validación: la ruta debe ser un texto no vacío
    if not isinstance(ruta, str) or not ruta.strip():
        raise ValueError("La ruta de la imagen debe ser un texto no vacío.")
    ruta = ruta.strip()

    # Caso 1: es una URL -> se descarga la imagen
    if ruta.lower().startswith(("http://", "https://")):
        try:
            respuesta = requests.get(ruta, headers=_HEADERS_HTTP, timeout=20)
            respuesta.raise_for_status()
        except requests.RequestException as error:
            raise FotoAppError(f"No se pudo descargar la imagen desde la URL: {error}") from error
        origen = BytesIO(respuesta.content)
    # Caso 2: es un archivo local o de Google Drive
    else:
        if not os.path.isfile(ruta):
            mensaje = f"No existe el archivo '{ruta}'."
            if ruta.startswith("/content/drive"):
                mensaje += (" ¿Montaste Google Drive? (from google.colab import drive; "
                            "drive.mount('/content/drive'))")
            raise FileNotFoundError(mensaje)
        origen = ruta

    # Abre la imagen, corrige la rotación y la convierte a RGB (3 canales)
    try:
        imagen = Image.open(origen)
        imagen = ImageOps.exif_transpose(imagen)  # respeta la rotación de la cámara
        return imagen.convert("RGB")
    except UnidentifiedImageError as error:
        raise FotoAppError(f"'{ruta}' no es un archivo de imagen válido.") from error


def normalizar_red(red_social):
    """Convierte la palabra clave a la clave interna ('Instagram ' -> 'instagram')."""
    if not isinstance(red_social, str):
        raise RedSocialInvalidaError("La red social debe ser un texto.")
    clave = red_social.strip().lower()
    clave = ALIAS_REDES.get(clave, clave)
    if clave not in DIMENSIONES_REDES:
        opciones = ", ".join(r.capitalize() for r in DIMENSIONES_REDES)
        raise RedSocialInvalidaError(
            f"Red social '{red_social}' no soportada. Opciones: {opciones}."
        )
    return clave


# ---------------------------------------------------------------------------
# 1. Leer y redimensionar para una red social
# ---------------------------------------------------------------------------


def redimensionar_para_red(ruta, red_social, modo="recortar", ruta_salida=None, mostrar=False):
    """Lee una foto y la redimensiona al tamaño recomendado de una red social.

    Parámetros
    ----------
    ruta : str
        Ruta local, ruta de Google Drive (p. ej. '/content/drive/MyDrive/foto.jpg')
        o URL de la imagen.
    red_social : str
        'Youtube', 'Instagram', 'Twitter' o 'Facebook' (no distingue mayúsculas).
    modo : str
        Cómo se respeta la escala para no distorsionar la foto:
        - 'recortar' (por defecto): escala la foto hasta cubrir el tamaño destino
          manteniendo la proporción y recorta lo que sobra, centrado.
        - 'ajustar': escala la foto hasta que entre completa y rellena con bandas
          blancas el espacio sobrante (no se pierde nada de la foto).
    ruta_salida : str, opcional
        Si se indica, guarda la imagen redimensionada en ese archivo.
    mostrar : bool
        Si es True, muestra la original y la redimensionada lado a lado.

    Devuelve
    --------
    PIL.Image.Image con exactamente el tamaño de la red social.
    """
    # Obtiene la red social y su tamaño (ancho, alto) recomendado
    clave = normalizar_red(red_social)
    tamano = DIMENSIONES_REDES[clave]
    original = leer_imagen(ruta)

    # Redimensiona sin deformar, según el modo elegido
    modo = modo.strip().lower()
    if modo == "recortar":
        resultado = ImageOps.fit(original, tamano, method=Image.LANCZOS, centering=(0.5, 0.5))
    elif modo == "ajustar":
        resultado = ImageOps.pad(original, tamano, method=Image.LANCZOS, color=(255, 255, 255))
    else:
        raise ValueError("modo debe ser 'recortar' o 'ajustar'.")

    # Guarda la imagen resultante si se indicó una ruta
    if ruta_salida:
        _crear_carpeta_de(ruta_salida)
        resultado.save(ruta_salida)

    # Muestra original y redimensionada lado a lado
    if mostrar:
        fig, ejes = plt.subplots(1, 2, figsize=(12, 5))
        ejes[0].imshow(original)
        ejes[0].set_title(f"Original {original.width}x{original.height}")
        ejes[1].imshow(resultado)
        ejes[1].set_title(f"{clave.capitalize()} {tamano[0]}x{tamano[1]} ({modo})")
        for eje in ejes:
            eje.axis("off")
        plt.tight_layout()
        plt.show()

    return resultado


# ---------------------------------------------------------------------------
# 2. Ecualización de histograma
# ---------------------------------------------------------------------------


def _ecualizar_canal(canal):
    """Ecualiza un canal de 8 bits (array 2D uint8) usando su histograma.

    1. Se calcula el histograma (cuántos píxeles hay de cada nivel 0..255).
    2. Se calcula la distribución acumulada (CDF).
    3. Se reescala la CDF a 0..255 y se usa como tabla de conversión: los
       niveles muy poblados se "separan" y el histograma queda más repartido,
       lo que aumenta el contraste.
    """
    histograma = np.bincount(canal.ravel(), minlength=256)  # píxeles por nivel
    cdf = histograma.cumsum()  # distribución acumulada
    cdf_min = cdf[cdf > 0][0]  # primer valor acumulado distinto de 0
    total = canal.size  # total de píxeles
    if total == cdf_min:  # imagen de un solo color: no hay nada que ecualizar
        return canal.copy()
    # Fórmula de ecualización: reescala la CDF al rango 0..255
    tabla = np.round((cdf - cdf_min) / (total - cdf_min) * 255)
    tabla = np.clip(tabla, 0, 255).astype(np.uint8)
    return tabla[canal]  # reemplaza cada píxel por su nuevo valor


def ecualizar_imagen(imagen):
    """Devuelve una copia de la imagen con el contraste ecualizado.

    En fotos color se ecualiza solo la luminancia (canal Y del espacio YCbCr)
    para no alterar los colores; ecualizar R, G y B por separado cambiaría
    los tonos.
    """
    _validar_imagen(imagen)
    # Imagen en grises: se ecualiza directamente
    if imagen.mode == "L":
        return Image.fromarray(_ecualizar_canal(np.asarray(imagen)))
    # Imagen color: se separa luminancia (Y) y color (Cb, Cr), se ecualiza solo Y y se recompone
    y, cb, cr = imagen.convert("YCbCr").split()
    y_eq = Image.fromarray(_ecualizar_canal(np.asarray(y)))
    return Image.merge("YCbCr", (y_eq, cb, cr)).convert("RGB")


def _graficar_histograma(eje, imagen, titulo):
    # Dibuja el histograma (barras) y la CDF (línea roja) de la imagen en grises
    luminancia = np.asarray(imagen.convert("L"))
    histograma = np.bincount(luminancia.ravel(), minlength=256)
    eje.bar(range(256), histograma, width=1.0, color="gray")
    eje.set_xlim(0, 255)
    # Si hay muchos píxeles quemados (255) o negros (0), ese pico taparía al
    # resto del histograma: limitamos el eje Y según los niveles intermedios.
    pico_intermedio = histograma[1:255].max()
    if pico_intermedio > 0 and histograma.max() > 3 * pico_intermedio:
        eje.set_ylim(0, pico_intermedio * 1.15)
        eje.text(0.5, 0.92, "(eje Y recortado: pico en 0 o 255)", transform=eje.transAxes,
                ha="center", fontsize=8, color="dimgray")
    eje.set_title(titulo)
    eje.set_xlabel("Nivel de gris")
    eje.set_ylabel("Cantidad de píxeles")
    cdf = histograma.cumsum()
    eje_cdf = eje.twinx()  # segundo eje Y para la CDF
    eje_cdf.plot(range(256), cdf / cdf[-1], color="tab:red", linewidth=1.5)
    eje_cdf.set_ylim(0, 1.05)
    eje_cdf.set_ylabel("Acumulado (CDF)", color="tab:red")


def ecualizar_histograma(imagen, ruta_figura=None, mostrar=True, titulo=""):
    """Ajusta el contraste de la foto con su histograma y grafica el resultado.

    Muestra en una misma figura la foto original y la ecualizada, cada una con
    su histograma, y guarda la figura si se indica `ruta_figura`.

    Devuelve la imagen ecualizada (PIL.Image.Image).
    """
    _validar_imagen(imagen)
    ecualizada = ecualizar_imagen(imagen)  # imagen con contraste ajustado

    # Figura 2x2: arriba las fotos, abajo sus histogramas
    fig, ejes = plt.subplots(2, 2, figsize=(13, 9), gridspec_kw={"height_ratios": [3, 2]})
    ejes[0, 0].imshow(imagen, cmap="gray")
    ejes[0, 0].set_title("Original")
    ejes[0, 1].imshow(ecualizada, cmap="gray")
    ejes[0, 1].set_title("Ecualizada")
    for eje in ejes[0]:
        eje.axis("off")
    _graficar_histograma(ejes[1, 0], imagen, "Histograma original")
    _graficar_histograma(ejes[1, 1], ecualizada, "Histograma ecualizado")
    fig.suptitle(f"Ecualización de histograma {('- ' + titulo) if titulo else ''}", fontsize=14)
    fig.tight_layout()
    _guardar_figura(fig, ruta_figura, mostrar)
    return ecualizada


# ---------------------------------------------------------------------------
# 3. Filtros de Pillow
# ---------------------------------------------------------------------------


def listar_filtros():
    """Devuelve un texto con los 9 filtros numerados (útil para el menú)."""
    return "\n".join(f"  {numero}. {nombre}" for numero, (nombre, _) in FILTROS.items())


def obtener_filtro(filtro):
    """Acepta el número (1-9) o el nombre del filtro y devuelve (nombre, filtro_pillow)."""
    if isinstance(filtro, str) and filtro.strip().isdigit():
        filtro = int(filtro.strip())
    if isinstance(filtro, int):
        if filtro in FILTROS:
            return FILTROS[filtro]
    elif isinstance(filtro, str):
        buscado = filtro.strip().lower().replace("_", " ")
        for nombre, filtro_pillow in FILTROS.values():
            if nombre.lower() == buscado:
                return nombre, filtro_pillow
    raise FiltroInvalidoError(
        f"Filtro '{filtro}' inválido. Elegí un número o nombre de:\n{listar_filtros()}"
    )


def aplicar_filtro(imagen, filtro, ruta_figura=None, mostrar=True):
    """Aplica uno de los 9 filtros de Pillow a la imagen.

    `filtro` puede ser el nombre ('Emboss', 'edge enhance', ...) o el número
    elegido por el usuario (1-9). Muestra la original y la filtrada lado a lado
    (la segunda con el nombre del filtro como título) y guarda la figura si se
    indica `ruta_figura`.

    Devuelve la imagen filtrada.
    """
    _validar_imagen(imagen)
    nombre, filtro_pillow = obtener_filtro(filtro)
    filtrada = imagen.filter(filtro_pillow)  # aplica el filtro de Pillow

    # Muestra original y filtrada lado a lado
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


def _impar(valor):
    """OpenCV exige tamaños de kernel impares: 4 -> 5, 5 -> 5."""
    valor = int(valor)
    return valor if valor % 2 == 1 else valor + 1


def boceto(imagen, tecnica="canny", grosor=1, **parametros):
    """Devuelve un boceto de líneas negras sobre fondo blanco (imagen en grises).

    Técnicas disponibles (y sus parámetros, ver TECNICAS_BOCETO):
    - 'umbral'     : umbralización adaptativa. Cada píxel se compara con el
                     promedio de su vecindario (`bloque`) menos `c`.
    - 'sobel'      : magnitud del gradiente de Sobel (derivadas en x e y) y
                     umbral `umbral` para quedarnos con los bordes fuertes.
    - 'laplaciano' : segunda derivada (Laplaciano) y umbral `umbral`.
    - 'canny'      : detector de Canny con histéresis (`umbral_bajo`,
                     `umbral_alto`). Da líneas finas y continuas.
    - 'lapiz'      : efecto "lápiz" (división de la imagen por su negativo
                     desenfocado). No es binaria: realza mucho los bordes.

    En todas, `blur` es el tamaño del desenfoque gaussiano previo, que elimina
    ruido y textura fina. `grosor` > 1 engrosa las líneas (dilatación).
    """
    _validar_imagen(imagen)
    tecnica = tecnica.strip().lower()
    if tecnica not in TECNICAS_BOCETO:
        raise TecnicaInvalidaError(
            f"Técnica '{tecnica}' inválida. Opciones: {', '.join(TECNICAS_BOCETO)}."
        )
    # Parámetros por defecto, sobrescritos por los que pase el usuario
    p = {**TECNICAS_BOCETO[tecnica], **parametros}

    # Pasa a grises y suaviza con desenfoque gaussiano para quitar ruido
    gris = np.asarray(imagen.convert("L"))
    suave = cv2.GaussianBlur(gris, (_impar(p["blur"]), _impar(p["blur"])), 0)

    if tecnica == "umbral":
        resultado = cv2.adaptiveThreshold(
            suave, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            _impar(p["bloque"]), p["c"],
        )
        bordes = resultado == 0
    elif tecnica == "sobel":
        gx = cv2.Sobel(suave, cv2.CV_64F, 1, 0, ksize=_impar(p["ksize"]))
        gy = cv2.Sobel(suave, cv2.CV_64F, 0, 1, ksize=_impar(p["ksize"]))
        magnitud = np.hypot(gx, gy)
        magnitud = magnitud / magnitud.max() * 255 if magnitud.max() > 0 else magnitud
        bordes = magnitud > p["umbral"]
    elif tecnica == "laplaciano":
        lap = np.abs(cv2.Laplacian(suave, cv2.CV_64F, ksize=_impar(p["ksize"])))
        lap = lap / lap.max() * 255 if lap.max() > 0 else lap
        bordes = lap > p["umbral"]
    elif tecnica == "canny":
        bordes = cv2.Canny(suave, p["umbral_bajo"], p["umbral_alto"]) > 0
    else:  # lapiz
        negativo_suave = cv2.GaussianBlur(255 - gris, (_impar(p["blur"]), _impar(p["blur"])), 0)
        lapiz = cv2.divide(gris, 255 - negativo_suave, scale=256)
        return Image.fromarray(lapiz)

    # Engrosa las líneas con dilatación si se pidió
    if grosor > 1:
        nucleo = np.ones((int(grosor), int(grosor)), np.uint8)
        bordes = cv2.dilate(bordes.astype(np.uint8), nucleo) > 0

    salida = np.where(bordes, 0, 255).astype(np.uint8)  # líneas negras, fondo blanco
    return Image.fromarray(salida)


def comparar_bocetos(imagen, configuraciones, ruta_figura=None, mostrar=True, columnas=3):
    """Muestra lado a lado la imagen original y varios bocetos.

    `configuraciones` es una lista de diccionarios, por ejemplo:
        [{"tecnica": "canny", "umbral_bajo": 50, "umbral_alto": 150},
         {"tecnica": "sobel", "umbral": 60}]

    Devuelve la lista de bocetos generados.
    """
    _validar_imagen(imagen)
    bocetos = []
    # Calcula la grilla: original + un boceto por configuración
    total = len(configuraciones) + 1
    filas = int(np.ceil(total / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(5 * columnas, 5 * filas))
    ejes = np.atleast_1d(ejes).ravel()
    ejes[0].imshow(imagen)
    ejes[0].set_title("Original")
    # Genera y dibuja cada boceto con su técnica y parámetros
    for eje, config in zip(ejes[1:], configuraciones):
        config = dict(config)
        tecnica = config.pop("tecnica")
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
    # Hace una pregunta s/n y devuelve True si la respuesta es afirmativa
    return entrada(pregunta + " (s/n): ").strip().lower() in ("s", "si", "sí", "y", "yes")


def menu(entrada=input, carpeta_salida=CARPETA_SALIDA, mostrar=True):
    """Menú interactivo de fotoapp.

    Para todas las opciones de edición (2 a 5) es necesario haber leído antes
    una imagen con la opción 1. Si el usuario intenta saltearse ese paso, las
    funciones lanzan ImagenNoCargadaError y el menú le explica qué hacer.

    `entrada` es la función que se usa para leer lo que escribe el usuario
    (por defecto `input`). En el notebook se reemplaza por respuestas
    simuladas para poder probar el menú de forma automática.
    """
    # Estado del menú: imagen en edición, su descripción y un contador para nombrar archivos
    imagen_actual = None
    descripcion = "ninguna (usá la opción 1)"
    contador = 0

    def ruta_nueva(nombre):
        # Genera un nombre de archivo único y numerado en la carpeta de salida
        nonlocal contador
        contador += 1
        return os.path.join(carpeta_salida, f"menu_{contador:02d}_{nombre}.png")

    # Bucle principal: se repite hasta elegir 0 (Salir)
    while True:
        print(OPCIONES_MENU.format(estado=descripcion))
        try:
            opcion = entrada("Elegí una opción: ").strip()

            if opcion == "0":
                print("¡Hasta luego!")
                break

            # Opción 1: leer la imagen y redimensionarla para una red social
            elif opcion == "1":
                ruta = entrada("Ruta (archivo, Drive) o URL de la imagen: ")
                red = entrada("Red social (Youtube / Instagram / Twitter / Facebook): ")
                modo = entrada("Modo [r]ecortar o [a]justar con bandas (Enter = recortar): ")
                modo = "ajustar" if modo.strip().lower().startswith("a") else "recortar"
                clave = normalizar_red(red)
                # La foto redimensionada se guarda en la carpeta de resultados
                nombre_base = os.path.splitext(os.path.basename(ruta.strip()))[0] or "foto"
                ruta_salida = os.path.join(carpeta_salida, f"{clave}_{nombre_base}.jpg")
                imagen_actual = redimensionar_para_red(ruta, red, modo=modo, ruta_salida=ruta_salida, mostrar=mostrar)
                descripcion = f"{os.path.basename(ruta.strip())} -> {clave.capitalize()} {imagen_actual.size}"
                print(f"Imagen lista para {clave.capitalize()}: {imagen_actual.size[0]}x{imagen_actual.size[1]} px")
                print(f"Imagen guardada en {ruta_salida}")

            # Opción 2: ecualizar histograma (requiere imagen cargada)
            elif opcion == "2":
                _validar_imagen(imagen_actual)
                ruta_figura = ruta_nueva("ecualizacion")
                resultado = ecualizar_histograma(imagen_actual, ruta_figura, mostrar=mostrar)
                print(f"Figura guardada en {ruta_figura}")
                if _preguntar_si(entrada, "¿Usar la imagen ecualizada como imagen actual?"):
                    imagen_actual = resultado
                    descripcion += " + ecualizada"

            # Opción 3: aplicar uno de los 9 filtros de Pillow
            elif opcion == "3":
                _validar_imagen(imagen_actual)
                print("Filtros disponibles:\n" + listar_filtros())
                nombre, _ = obtener_filtro(entrada("Número de filtro (1-9): "))
                ruta_figura = ruta_nueva("filtro_" + nombre.lower().replace(" ", "_"))
                resultado = aplicar_filtro(imagen_actual, nombre, ruta_figura, mostrar=mostrar)
                print(f"Figura guardada en {ruta_figura}")
                if _preguntar_si(entrada, f"¿Usar la imagen con {nombre} como imagen actual?"):
                    imagen_actual = resultado
                    descripcion += f" + {nombre}"

            # Opción 4: generar boceto con bordes y contornos
            elif opcion == "4":
                _validar_imagen(imagen_actual)
                tecnicas = list(TECNICAS_BOCETO)
                print("Técnicas:\n" + "\n".join(f"  {i}. {t}" for i, t in enumerate(tecnicas, 1)))
                eleccion = entrada("Número de técnica (Enter = canny): ").strip() or "4"
                if not eleccion.isdigit() or not 1 <= int(eleccion) <= len(tecnicas):
                    raise TecnicaInvalidaError(f"Opción de técnica '{eleccion}' inválida.")
                tecnica = tecnicas[int(eleccion) - 1]
                ruta_figura = ruta_nueva("boceto_" + tecnica)
                resultado = comparar_bocetos(
                    imagen_actual, [{"tecnica": tecnica}], ruta_figura, mostrar=mostrar, columnas=2
                )[0]
                print(f"Figura guardada en {ruta_figura}")
                if _preguntar_si(entrada, "¿Usar el boceto como imagen actual?"):
                    imagen_actual = resultado.convert("RGB")
                    descripcion += f" + boceto {tecnica}"

            # Opción 5: guardar la imagen actual en disco
            elif opcion == "5":
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

        # Manejo de errores: el menú informa el problema y sigue funcionando
        except ImagenNoCargadaError as error:
            print(f"[!] {error}")
        except (FotoAppError, FileNotFoundError, ValueError) as error:
            print(f"[!] Error: {error}")
        except (KeyboardInterrupt, EOFError):
            print("\nSaliendo de fotoapp.")
            break


def main():
    """Punto de entrada de la app: inicia el menú interactivo por consola."""
    menu()


# Solo se ejecuta el menú si se corre el archivo directamente (python fotoapp.py)
if __name__ == "__main__":
    main()
