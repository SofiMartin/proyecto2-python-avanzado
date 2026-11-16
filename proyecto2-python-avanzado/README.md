# Proyecto 2: Procesamiento de imágenes y modelos de regresión con Python

**Programación Python Avanzada — Tecnicatura Universitaria en Tecnologías de Programación — 2026**

Integrantes: Sofia Martin, Laura Sofía Pérez Llanqueleo

| Parte | Contenido | Abrir en Colab |
|---|---|---|
| 1. fotoapp | [`parte1_fotoapp/fotoapp.py`](parte1_fotoapp/fotoapp.py) (módulo) y [`fotoapp_testdev.ipynb`](parte1_fotoapp/fotoapp_testdev.ipynb) (pruebas) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SofiMartin/proyecto2-python-avanzado/blob/master/proyecto2-python-avanzado/parte1_fotoapp/fotoapp_testdev.ipynb) |

```
proyecto2-python-avanzado/
├── README.md
├── requirements.txt
├── parte1_fotoapp/
│   ├── fotoapp.py                 # módulo con todas las funciones + menú
│   ├── fotoapp_testdev.ipynb      # notebook de pruebas
│   ├── fotos_mejoradas/           # resultados del menú (fotos redimensionadas y figuras)
│   └── imagenes/
│       ├── originales/            # fotos de prueba (+ CREDITOS.md)
│       └── generadas/             # figuras e imágenes generadas
```

## Cómo ejecutarlo

- **Google Colab:** abrir los notebooks con los botones de arriba y ejecutar todas las celdas. La primera celda clona este repositorio para tener `fotoapp.py`, las imágenes y el dataset. Opcionalmente se puede montar Google Drive para usar fotos propias.
- **Jupyter local:**
  ```bash
  pip install -r requirements.txt
  jupyter notebook
  ```
- **Menú por consola:**
  ```bash
  cd parte1_fotoapp
  python fotoapp.py
  ```

---

## Parte 1: fotoapp

| # | Función | Qué hace |
|---|---|---|
| 1 | `redimensionar_para_red(ruta, red_social, modo="recortar")` | Lee la foto (archivo local, Google Drive o URL) y la lleva al tamaño recomendado de **YouTube (1280×720)**, **Instagram (1080×1080)**, **Twitter/X (1600×900)** o **Facebook (1200×630)** sin distorsionarla. Lo hace recortando centrado (`resize` + `crop`) o ajustando con bandas blancas (`resize` + `paste`). |
| 2 | `ecualizar_histograma(imagen, ruta_figura)` | Ecualiza el histograma con `cv2.equalizeHist` (en fotos color, solo el brillo: canal Y de YCrCb) y muestra/guarda una figura con la foto original, la ecualizada y sus histogramas. Probada con una foto oscura y otra con exceso de luz. |
| 3 | `aplicar_filtro(imagen, filtro, ruta_figura)` | Aplica uno de los 9 filtros de Pillow (por número 1-9 o por nombre) y muestra/guarda la original y la filtrada lado a lado. |
| 4 | `boceto(imagen, tecnica, **parametros)` y `comparar_bocetos(...)` | Boceto de líneas negras sobre fondo blanco con **umbral adaptativo**, **Sobel**, **Laplaciano** o **Canny**. Parámetros ajustables y comparación lado a lado. |
| 5 | `main()` / `menu()` | Menú por consola. Obliga a pasar primero por la función 1 y valida todo con excepciones. Guarda todos los resultados en la carpeta `fotos_mejoradas`. |

**Excepciones propias:** `ImagenNoCargadaError`, `RedSocialInvalidaError`, `FiltroInvalidoError` y `TecnicaInvalidaError`. Todas las funciones de edición llaman a `_validar_imagen()`, que lanza `ImagenNoCargadaError` si todavía no se leyó ninguna imagen. Así el usuario no puede saltearse la lectura.

### Diagrama de flujo del menú

```mermaid
flowchart TD
    A([Inicio: main → menu]) --> B["imagen_actual = None<br/>contador = 0"]
    B --> C[/"Mostrar menú con<br/>estado de la imagen actual"/]
    C --> D[/"Leer opción"/]
    D --> E{¿Qué opción?}

    E -->|0| Z["Imprimir '¡Hasta luego!'"] --> FIN([Fin])
    E -->|Otra| INV["Imprimir 'Opción inválida.<br/>Escribí un número del 0 al 5'"] --> C

    %% Opción 1
    E -->|1| O1A[/"Pedir ruta o URL,<br/>red social y modo"/]
    O1A --> O1B["normalizar_red<br/>(acepta alias: x, ig, fb, yt)"]
    O1B --> O1C{¿Red social válida?}
    O1C -->|No| ERR1["RedSocialInvalidaError"]
    O1C -->|Sí| O1D{"¿Se puede leer la imagen?<br/>(archivo, Drive o URL)"}
    O1D -->|"No: archivo inexistente,<br/>URL caída o no es imagen"| ERR2["FileNotFoundError / ValueError / OSError"]
    O1D -->|Sí| O1E["Redimensionar sin distorsión<br/>(recortar o ajustar con bandas)"]
    O1E --> O1F["Guardar en carpeta de salida<br/>imagen_actual = resultado"]
    O1F --> C

    %% Opciones 2 a 5: validación previa
    E -->|"2, 3, 4 o 5"| V{"¿Hay imagen_actual<br/>cargada?"}
    V -->|No| ERR3["ImagenNoCargadaError:<br/>'Primero leé la foto con la opción 1'"]
    V -->|Sí| S{¿Cuál de las cuatro?}

    %% Opción 2
    S -->|2| O2A["Ecualizar histograma<br/>Guardar y mostrar figura"]
    O2A --> U{"¿Usar resultado como<br/>imagen actual?"}

    %% Opción 3
    S -->|3| O3A[/"Mostrar filtros y pedir número 1-9"/]
    O3A --> O3B{¿Filtro válido?}
    O3B -->|No| ERR4["FiltroInvalidoError"]
    O3B -->|Sí| O3C["Aplicar filtro de Pillow<br/>Guardar y mostrar figura"]
    O3C --> U

    %% Opción 4
    S -->|4| O4A[/"Mostrar técnicas y pedir número<br/>(Enter = canny)"/]
    O4A --> O4B{¿Técnica válida?}
    O4B -->|No| ERR5["TecnicaInvalidaError"]
    O4B -->|Sí| O4C["Generar boceto<br/>Guardar y mostrar figura"]
    O4C --> U

    %% Uso del resultado
    U -->|Sí| U1["imagen_actual = resultado<br/>Actualizar descripción"] --> C
    U -->|No| C

    %% Opción 5
    S -->|5| O5A[/"Pedir nombre de archivo<br/>(Enter = automático)"/]
    O5A --> O5B["Agregar .png si no tiene extensión<br/>Guardar imagen_actual"]
    O5B --> C

    %% Manejo de errores
    ERR1 --> MSG["Imprimir '[!] Error: ...'<br/>o '[!] mensaje'"]
    ERR2 --> MSG
    ERR3 --> MSG
    ERR4 --> MSG
    ERR5 --> MSG
    MSG --> C

    %% Interrupción
    D -.->|"Ctrl+C o fin de entrada"| SAL["Imprimir 'Saliendo de fotoapp'"] -.-> FIN
```

### Resultados destacados

| Ecualización: foto oscura | Ecualización: exceso de luz |
|---|---|
| ![](parte1_fotoapp/imagenes/generadas/2_ecualizacion_foto_oscura.png) | ![](parte1_fotoapp/imagenes/generadas/2_ecualizacion_foto_con_exceso_de_luz.png) |

![Comparativa de los 9 filtros](parte1_fotoapp/imagenes/generadas/3_comparativa_9_filtros.png)

![Comparativa de técnicas de boceto](parte1_fotoapp/imagenes/generadas/4_comparativa_tecnicas.png)

Las fotos de prueba son de Wikimedia Commons. Ver [créditos y licencias](parte1_fotoapp/imagenes/originales/CREDITOS.md).
