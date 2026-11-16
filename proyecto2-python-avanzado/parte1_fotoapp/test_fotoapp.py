"""Tests de fotoapp.py. Ejecutar con:  pytest  (desde parte1_fotoapp)"""

import matplotlib

matplotlib.use("Agg")  # sin ventanas durante los tests

import fotoapp
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def foto(tmp_path):
    """Foto de prueba de 400x300 con un degradado."""
    arreglo = np.zeros((300, 400, 3), dtype=np.uint8)
    arreglo[:, :, 0] = np.linspace(0, 255, 400, dtype=np.uint8)
    arreglo[:, :, 1] = 100
    arreglo[100:200, 150:250] = 255   # cuadrado blanco: bordes bien marcados
    ruta = tmp_path / "foto.png"
    Image.fromarray(arreglo).save(ruta)
    return str(ruta)


@pytest.fixture
def imagen(foto):
    return fotoapp.leer_imagen(foto)


# --- 1. redimensionar_para_red -------------------------------------------

@pytest.mark.parametrize("red", ["Youtube", "Instagram", "Twitter", "Facebook"])
@pytest.mark.parametrize("modo", ["recortar", "ajustar"])
def test_tamano_exacto_por_red(foto, red, modo):
    resultado = fotoapp.redimensionar_para_red(foto, red, modo=modo)
    assert resultado.size == fotoapp.DIMENSIONES_REDES[red.lower()]


def test_guarda_archivo_de_salida(foto, tmp_path):
    salida = tmp_path / "sub" / "out.jpg"
    fotoapp.redimensionar_para_red(foto, "x", ruta_salida=str(salida))
    assert salida.exists()


def test_ajustar_rellena_con_blanco(foto):
    resultado = fotoapp.redimensionar_para_red(foto, "Instagram", modo="ajustar")
    assert resultado.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.parametrize("alias,esperado", [("  INSTAGRAM ", "instagram"), ("x", "twitter"),
                                            ("Fb", "facebook"), ("yt", "youtube")])
def test_normalizar_red(alias, esperado):
    assert fotoapp.normalizar_red(alias) == esperado


def test_red_invalida(foto):
    with pytest.raises(fotoapp.RedSocialInvalidaError):
        fotoapp.redimensionar_para_red(foto, "TikTok")
    with pytest.raises(fotoapp.RedSocialInvalidaError):
        fotoapp.normalizar_red(5)


def test_modo_invalido(foto):
    with pytest.raises(ValueError):
        fotoapp.redimensionar_para_red(foto, "Instagram", modo="estirar")
    with pytest.raises(TypeError):
        fotoapp.redimensionar_para_red(foto, "Instagram", modo=None)


def test_archivo_inexistente():
    with pytest.raises(FileNotFoundError):
        fotoapp.leer_imagen("no_existe.jpg")


def test_drive_sin_montar_sugiere_montar():
    with pytest.raises(FileNotFoundError, match="drive.mount"):
        fotoapp.leer_imagen("/content/drive/MyDrive/foto.jpg")


def test_archivo_que_no_es_imagen(tmp_path):
    texto = tmp_path / "nota.txt"
    texto.write_text("hola")
    with pytest.raises(ValueError):
        fotoapp.leer_imagen(str(texto))


@pytest.mark.parametrize("ruta", ["", "   ", None, 3])
def test_ruta_vacia_o_invalida(ruta):
    with pytest.raises(ValueError):
        fotoapp.leer_imagen(ruta)


def test_png_transparente_queda_con_fondo_blanco(tmp_path):
    ruta = tmp_path / "transparente.png"
    Image.new("RGBA", (50, 50), (0, 0, 0, 0)).save(ruta)
    assert fotoapp.leer_imagen(str(ruta)).getpixel((10, 10)) == (255, 255, 255)


# --- 2. ecualización -------------------------------------------------------

def test_ecualizar_aumenta_contraste(imagen):
    oscura = Image.fromarray((np.array(imagen) * 0.2).astype(np.uint8))
    ecualizada = fotoapp.ecualizar_histograma(oscura, mostrar=False)
    assert ecualizada.size == oscura.size
    assert np.array(ecualizada.convert("L")).std() > np.array(oscura.convert("L")).std()


@pytest.mark.parametrize("metodo", ["global", "clahe", " CLAHE "])
def test_ecualizar_metodos(imagen, metodo):
    oscura = Image.fromarray((np.array(imagen) * 0.2).astype(np.uint8))
    ecualizada = fotoapp.ecualizar_imagen(oscura, metodo)
    assert ecualizada.size == oscura.size
    assert np.array(ecualizada.convert("L")).std() > np.array(oscura.convert("L")).std()


def test_clahe_distinto_de_global(imagen):
    assert np.array(fotoapp.ecualizar_imagen(imagen, "clahe")).tolist() != \
        np.array(fotoapp.ecualizar_imagen(imagen, "global")).tolist()


@pytest.mark.parametrize("metodo", ["otro", None, 3])
def test_ecualizar_metodo_invalido(imagen, metodo):
    with pytest.raises(ValueError):
        fotoapp.ecualizar_imagen(imagen, metodo)


def test_ecualizar_guarda_figura(imagen, tmp_path):
    ruta = tmp_path / "figuras" / "eq.png"
    fotoapp.ecualizar_histograma(imagen, str(ruta), mostrar=False)
    assert ruta.exists()


# --- 3. filtros -------------------------------------------------------------

def test_los_9_filtros(imagen):
    assert len(fotoapp.FILTROS) == 9
    for numero in fotoapp.FILTROS:
        filtrada = fotoapp.aplicar_filtro(imagen, numero, mostrar=False)
        assert filtrada.size == imagen.size


def test_filtro_por_nombre_y_numero_texto(imagen):
    assert fotoapp.obtener_filtro("emboss")[0] == "Emboss"
    assert fotoapp.obtener_filtro("8")[0] == "Sharpen"
    assert fotoapp.obtener_filtro("edge_enhance")[0] == "Edge Enhance"


@pytest.mark.parametrize("filtro", [0, 10, "nada", None])
def test_filtro_invalido(imagen, filtro):
    with pytest.raises(fotoapp.FiltroInvalidoError):
        fotoapp.aplicar_filtro(imagen, filtro, mostrar=False)


# --- 4. bocetos --------------------------------------------------------------

@pytest.mark.parametrize("tecnica", list(fotoapp.TECNICAS_BOCETO))
def test_boceto_es_blanco_y_negro(imagen, tecnica):
    resultado = fotoapp.boceto(imagen, tecnica)
    assert resultado.mode == "L" and resultado.size == imagen.size
    assert set(np.unique(np.array(resultado))) <= {0, 255}


def test_boceto_grosor_agrega_lineas(imagen):
    fino = np.array(fotoapp.boceto(imagen, "canny", grosor=1))
    grueso = np.array(fotoapp.boceto(imagen, "canny", grosor=3))
    assert (grueso == 0).sum() > (fino == 0).sum()


def test_tecnica_invalida(imagen):
    with pytest.raises(fotoapp.TecnicaInvalidaError):
        fotoapp.boceto(imagen, "inexistente")


def test_comparar_bocetos(imagen):
    bocetos = fotoapp.comparar_bocetos(imagen, [{"tecnica": "canny"}, {"tecnica": "sobel", "umbral": 40}],
                                       mostrar=False)
    assert len(bocetos) == 2


# --- Imagen no cargada --------------------------------------------------------

@pytest.mark.parametrize("funcion,args", [
    (fotoapp.ecualizar_histograma, ()),
    (fotoapp.aplicar_filtro, (1,)),
    (fotoapp.boceto, ()),
    (fotoapp.comparar_bocetos, ([],)),
])
@pytest.mark.parametrize("sin_imagen", [None, "foto.jpg", 42])
def test_editar_sin_imagen_cargada(funcion, args, sin_imagen):
    with pytest.raises(fotoapp.ImagenNoCargadaError):
        funcion(sin_imagen, *args)


# --- 5. menú -------------------------------------------------------------------

def correr_menu(respuestas, carpeta, capsys):
    iterador = iter(respuestas)

    def entrada(_=""):
        try:
            return next(iterador)
        except StopIteration:
            raise EOFError from None

    fotoapp.menu(entrada=entrada, carpeta_salida=str(carpeta), mostrar=False)
    return capsys.readouterr().out


def test_menu_obliga_a_leer_primero(tmp_path, capsys):
    salida = correr_menu(["2", "3", "4", "5", "0"], tmp_path, capsys)
    assert salida.count("Primero leé la foto con la opción 1") == 4


def test_menu_flujo_completo(foto, tmp_path, capsys):
    salida = correr_menu([
        "1", foto, "yt", "",     # leer y redimensionar
        "2", "c", "s",           # ecualizar con CLAHE y usar resultado
        "3", "6", "n",           # filtro Emboss
        "4", "", "n",            # boceto canny
        "5", "final",            # guardar
        "0",
    ], tmp_path, capsys)
    assert "Imagen lista para Youtube: 1280x720 px" in salida
    assert (tmp_path / "final.png").exists()
    assert (tmp_path / "youtube_foto.jpg").exists()
    assert "[!]" not in salida


def test_menu_errores_no_lo_cierran(foto, tmp_path, capsys):
    salida = correr_menu(["9", "1", "no_existe.jpg", "ig", "", "1", foto, "tiktok", "", "0"], tmp_path, capsys)
    assert "Opción inválida" in salida
    assert "No existe el archivo" in salida
    assert "no soportada" in salida
    assert "¡Hasta luego!" in salida


def test_menu_termina_con_fin_de_entrada(tmp_path, capsys):
    assert "Saliendo de fotoapp" in correr_menu([], tmp_path, capsys)
