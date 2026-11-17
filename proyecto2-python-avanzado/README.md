# Proyecto 2: fotoapp, procesamiento de imágenes con Python

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
| 2 | `ecualizar_histograma(imagen, ruta_figura, metodo="global")` | Ecualiza el histograma con `cv2.equalizeHist` (`metodo="global"`) o con `cv2.createCLAHE` (`metodo="clahe"`, adaptativo por zonas, evita el exceso de contraste). En fotos color, solo el brillo: canal Y de YCrCb y muestra/guarda una figura con la foto original, la ecualizada y sus histogramas. Probada con una foto oscura y otra con exceso de luz. |
| 3 | `aplicar_filtro(imagen, filtro, ruta_figura)` | Aplica uno de los 9 filtros de Pillow (por número 1-9 o por nombre) y muestra/guarda la original y la filtrada lado a lado. |
| 4 | `boceto(imagen, tecnica, **parametros)` y `comparar_bocetos(...)` | Boceto de líneas negras sobre fondo blanco con **umbral adaptativo**, **Sobel**, **Laplaciano** o **Canny**. Parámetros ajustables y comparación lado a lado. |
| 5 | `main()` / `menu()` | Menú por consola. Obliga a pasar primero por la función 1 y valida todo con excepciones. Guarda todos los resultados en la carpeta `fotos_mejoradas`. |

**Imágenes con transparencia:** si la foto es un PNG con transparencia, se pega sobre fondo blanco al leerla (si no, el fondo quedaría negro).

**Mostrar figuras:** el parámetro `mostrar` de las funciones es `None` por defecto: la figura se muestra solo en Jupyter/Colab. Con `True` se muestra siempre y con `False` nunca, por ejemplo al importar el módulo desde un script.

**Excepciones propias:** `ImagenNoCargadaError`, `RedSocialInvalidaError`, `FiltroInvalidoError` y `TecnicaInvalidaError`. Todas las funciones de edición llaman a `_validar_imagen()`, que lanza `ImagenNoCargadaError` si todavía no se leyó ninguna imagen. Así el usuario no puede saltearse la lectura.

### Diagrama de flujo del menú

El flujo se divide en tres diagramas para que se lea bien: el ciclo general del menú, la opción 1 y las opciones 2 a 5. Los colores indican el tipo de paso: verde = inicio o fin, azul = entrada del usuario, amarillo = decisión, gris = proceso, rojo = error o mensaje, violeta = detalle en otro diagrama.

#### 1. Ciclo general

```mermaid
flowchart TD
    INICIO([Inicio: main llama a menu]) --> INIT["imagen_actual = None<br/>contador de archivos = 0"]
    INIT --> MENU[/"Mostrar el menú con el estado<br/>de la imagen actual"/]
    MENU --> LEER[/"Leer opción"/]
    LEER --> OPCION{"¿Qué opción?"}
    LEER -.->|"Ctrl+C o fin de entrada"| SALIR["Mostrar 'Saliendo de fotoapp'"]

    OPCION -->|0| CHAU["Mostrar '¡Hasta luego!'"]
    OPCION -->|otra| INVALIDA["Mostrar 'Opción inválida.<br/>Escribí un número del 0 al 5'"]
    OPCION -->|1| OP1["Opción 1<br/>Leer y redimensionar<br/>(detalle en el diagrama 2)"]
    OPCION -->|"2, 3, 4 o 5"| VAL{"¿Hay imagen_actual<br/>cargada?"}
    VAL -->|Sí| OP25["Opciones 2 a 5<br/>Ecualizar, filtro, boceto o guardar<br/>(detalle en el diagrama 3)"]
    VAL -->|No| ERR3["ImagenNoCargadaError<br/>'Primero leé la foto con la opción 1'"]

    OP1 -->|"Sin errores"| VOLVER([Volver al menú])
    OP1 -->|"Con error"| ERR["Mostrar '[!] Error: ...'<br/>El programa no se cierra"]
    OP25 -->|"Sin errores"| VOLVER
    OP25 -->|"Con error"| ERR
    ERR3 --> VOLVER
    ERR --> VOLVER
    INVALIDA --> VOLVER
    VOLVER --> MENU

    CHAU --> FIN([Fin])
    SALIR -.-> FIN

    classDef terminal fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    classDef decision fill:#fff8e1,stroke:#f9a825,color:#5d4037
    classDef entrada fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef error fill:#ffebee,stroke:#c62828,color:#7f1d1d
    classDef proceso fill:#f5f5f5,stroke:#616161,color:#212121
    classDef detalle fill:#ede7f6,stroke:#5e35b1,color:#311b92

    class INICIO,FIN,VOLVER terminal
    class OPCION,VAL decision
    class MENU,LEER entrada
    class ERR,ERR3,INVALIDA,SALIR error
    class INIT,CHAU proceso
    class OP1,OP25 detalle
```

#### 2. Opción 1: leer y redimensionar

```mermaid
flowchart TD
    A[/"Pedir ruta o URL de la imagen,<br/>red social y modo<br/>(Enter = recortar)"/] --> B["normalizar_red<br/>(acepta alias: x, ig, fb, yt)"]
    B --> C{"¿Red social válida?"}
    C -->|No| E1["RedSocialInvalidaError"]
    C -->|Sí| D{"¿Se puede leer la imagen?<br/>(archivo, Drive o URL)"}
    D -->|"No: archivo inexistente,<br/>URL caída o no es imagen"| E2["FileNotFoundError / ValueError / OSError"]
    D -->|Sí| F["Redimensionar sin distorsión<br/>(recortar o ajustar con bandas)"]
    F --> G["Guardar en la carpeta de salida<br/>imagen_actual = resultado"]
    G --> VOLVER([Volver al menú])
    E1 --> MSG["Mostrar '[!] Error: ...'"]
    E2 --> MSG
    MSG --> VOLVER

    classDef terminal fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    classDef decision fill:#fff8e1,stroke:#f9a825,color:#5d4037
    classDef entrada fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef error fill:#ffebee,stroke:#c62828,color:#7f1d1d
    classDef proceso fill:#f5f5f5,stroke:#616161,color:#212121

    class VOLVER terminal
    class C,D decision
    class A entrada
    class E1,E2,MSG error
    class B,F,G proceso
```

#### 3. Opciones 2 a 5: ecualizar, filtro, boceto y guardar

```mermaid
flowchart TD
    START(["Hay una imagen cargada"]) --> ELEGIR{"¿Qué opción?"}

    ELEGIR -->|2| O2A[/"Pedir método<br/>(g = global, c = clahe, Enter = global)"/]
    O2A --> O2B["Ecualizar el histograma<br/>Guardar y mostrar la figura"]

    ELEGIR -->|3| O3A[/"Mostrar los 9 filtros<br/>y pedir el número"/]
    O3A --> O3B{"¿Filtro válido?"}
    O3B -->|No| E4["FiltroInvalidoError"]
    O3B -->|Sí| O3C["Aplicar el filtro de Pillow<br/>Guardar y mostrar la figura"]

    ELEGIR -->|4| O4A[/"Mostrar las técnicas y pedir el número<br/>(Enter = canny)"/]
    O4A --> O4B{"¿Técnica válida?"}
    O4B -->|No| E5["TecnicaInvalidaError"]
    O4B -->|Sí| O4C["Generar el boceto<br/>Guardar y mostrar la figura"]

    O2B --> USAR{"¿Usar el resultado como<br/>imagen actual?"}
    O3C --> USAR
    O4C --> USAR
    USAR -->|Sí| USAR_SI["imagen_actual = resultado<br/>Actualizar la descripción"]
    USAR -->|No| OK
    USAR_SI --> OK([Volver al menú])

    ELEGIR -->|5| O5A[/"Pedir nombre de archivo<br/>(Enter = automático)"/]
    O5A --> O5B["Agregar .png si no tiene extensión<br/>Guardar imagen_actual"]
    O5B --> OK

    E4 --> MSG["Mostrar '[!] Error: ...'"]
    E5 --> MSG
    MSG --> OK

    classDef terminal fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    classDef decision fill:#fff8e1,stroke:#f9a825,color:#5d4037
    classDef entrada fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef error fill:#ffebee,stroke:#c62828,color:#7f1d1d
    classDef proceso fill:#f5f5f5,stroke:#616161,color:#212121

    class START,OK terminal
    class ELEGIR,O3B,O4B,USAR decision
    class O2A,O3A,O4A,O5A entrada
    class E4,E5,MSG error
    class O2B,O3C,O4C,O5B,USAR_SI proceso
```

### Resultados destacados

| Ecualización: foto oscura | Ecualización: exceso de luz |
|---|---|
| ![](parte1_fotoapp/imagenes/generadas/2_ecualizacion_foto_oscura.png) | ![](parte1_fotoapp/imagenes/generadas/2_ecualizacion_foto_con_exceso_de_luz.png) |

![Comparativa de los 9 filtros](parte1_fotoapp/imagenes/generadas/3_comparativa_9_filtros.png)

![Comparativa de técnicas de boceto](parte1_fotoapp/imagenes/generadas/4_comparativa_tecnicas.png)

Las fotos de prueba son de Wikimedia Commons. Ver [créditos y licencias](parte1_fotoapp/imagenes/originales/CREDITOS.md).
