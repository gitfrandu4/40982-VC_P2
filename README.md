# Práctica 2: Funciones básicas de OpenCV

Este repositorio contiene la **Práctica 2** sobre el uso de **OpenCV** y **NumPy** para el procesamiento de imágenes, donde se exploran tareas como la conversión a escala de grises, la detección de bordes y el análisis de píxeles en imágenes.

# Índice

- [Práctica 2: Funciones básicas de OpenCV](#práctica-2-funciones-básicas-de-opencv)
- [Índice](#índice)
  - [Librerías utilizadas](#librerías-utilizadas)
  - [Autores](#autores)
  - [Tareas](#tareas)
    - [Tarea 1](#tarea-1)
      - [1.1 Detección de Bordes con Canny y Análisis de Píxeles](#11-detección-de-bordes-con-canny-y-análisis-de-píxeles)
      - [Tarea 1.2](#tarea-12)
    - [Tarea 2](#tarea-2)
      - [Procesamiento de la Imagen](#procesamiento-de-la-imagen)
      - [Análisis de Píxeles](#análisis-de-píxeles)
      - [Identificación de Filas y Columnas Significativas](#identificación-de-filas-y-columnas-significativas)
      - [Visualización](#visualización)
      - [Resultado](#resultado)
    - [Tarea 3](#tarea-3)
      - [Captura de la Cámara Web:](#captura-de-la-cámara-web)
      - [Efecto de Detección de Bordes (Esquina Superior Derecha):](#efecto-de-detección-de-bordes-esquina-superior-derecha)
      - [Detección de Contraste (Esquina Inferior Derecha):](#detección-de-contraste-esquina-inferior-derecha)
      - [Efecto Pop Art:](#efecto-pop-art)
      - [Combinación y Visualización:](#combinación-y-visualización)
      - [Resultado](#resultado-1)
    - [Tarea 4](#tarea-4)
      - [Descripción del Proyecto](#descripción-del-proyecto)
      - [Conceptos y Algoritmos Utilizados](#conceptos-y-algoritmos-utilizados)
      - [Visualización](#visualización-1)
      - [Conclusiones](#conclusiones)
  - [Referencias y bibliografía](#referencias-y-bibliografía)

## Librerías utilizadas

[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy)](https://numpy.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=matplotlib)](https://matplotlib.org/)

## Autores

Este proyecto fue desarrollado por:

- [![GitHub](https://img.shields.io/badge/GitHub-Francisco%20Javier%20L%C3%B3pez%E2%80%93Dufour%20Morales-yellow?style=flat-square&logo=github)](https://github.com/gitfrandu4)
- [![GitHub](https://img.shields.io/badge/GitHub-Marcos%20V%C3%A1zquez%20Tasc%C3%B3n-red?style=flat-square&logo=github)](https://github.com/DerKom)

## Tareas

### Tarea 1

#### 1.1 Detección de Bordes con Canny y Análisis de Píxeles

* **Canny**: Se convierte la imagen a escala de grises y se aplica el detector de bordes Canny con umbrales 250 y 100.
* **Análisis de Píxeles Blancos**: Se cuenta el número de píxeles blancos por filas y columnas usando `cv2.reduce` y `np.sum`. Los resultados se normalizan para obtener porcentajes.

<img src="Images/task_1_1.png" width="800" alt="Resultado Tarea 1.1">
  
#### Tarea 1.2

* **Cálculo de `maxfil`**: Se determina el número máximo de píxeles blancos por fila y columna (`maxfil` y `maxcol`).
* **Umbral 95%**: Se identifican filas y columnas con al menos el 95% de los píxeles blancos de `maxfil` y `maxcol`.
* **Resultado**: Se muestran las posiciones de las filas y columnas clave y se resaltan visualmente en la imagen.

```
* Maximum white pixels in a row: 218
* Number of rows with white pixels >= 0.95 * maxfil: 2
* Positions of these rows: [ 12 100]

* Maximum white pixels in a column: 184
* Number of columns with white pixels >= 0.95 * maxcol: 3
* Positions of these columns: [104 115 119]
```

<img src="Images/task_1_2.png" width="300" alt="Resultado Tarea 1.2">

### Tarea 2
En este proyecto, aplicamos el operador Sobel a una imagen en escala de grises para detectar sus bordes y luego binarizamos el resultado mediante umbralizado. Los pasos clave fueron:

#### Procesamiento de la Imagen

- Convertimos la imagen a escala de grises y aplicamos un filtro Gaussiano para suavizarla y reducir el ruido.
- Calculamos el gradiente Sobel en las direcciones x e y y combinamos ambos resultados.
- Convertimos el resultado a 8 bits y aplicamos un umbral para obtener una imagen binaria de los bordes.

```python
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ggris = cv2.GaussianBlur(gris, (3, 3), 0)
sobelx = cv2.Sobel(ggris, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(ggris, cv2.CV_64F, 0, 1)
sobel = cv2.add(sobelx, sobely)
sobel8 = cv2.convertScaleAbs(sobel)
_, sobel8Umbralizado = cv2.threshold(sobel8, 130, 255, cv2.THRESH_BINARY)
```

#### Análisis de Píxeles

- Contamos el número de píxeles blancos por filas y columnas utilizando `cv2.reduce`.
- Normalizamos estos conteos para obtener porcentajes respecto al total.

```python
col_counts = cv2.reduce(sobel8Umbralizado, 0, cv2.REDUCE_SUM)
fil_counts = cv2.reduce(sobel8Umbralizado, 1, cv2.REDUCE_SUM).flatten()
cols = col_counts[0] / (255 * sobel8Umbralizado.shape[0])
filas = fil_counts / (255 * sobel8Umbralizado.shape[1])
```

#### Identificación de Filas y Columnas Significativas

- Calculamos el 95% del valor máximo de los porcentajes obtenidos.
- Identificamos las filas y columnas que superan este umbral.

```python
limiteFila = np.max(filas) * 0.95
limiteColumna = np.max(cols) * 0.95
filas_superiores = np.where(filas >= limiteFila)[0]
columnas_superiores = np.where(cols >= limiteColumna)[0]
```

#### Visualización

- Remarcamos las filas y columnas significativas en la imagen binarizada.
- Observamos que, en comparación con el método de Canny, Sobel es más sensible al ruido y produce bordes menos definidos, mientras que Canny ofrece una detección más precisa.

```python
sobel8Umbralizado_bgr = cv2.cvtColor(sobel8Umbralizado, cv2.COLOR_GRAY2BGR)
for fila in filas_superiores:
    cv2.line(sobel8Umbralizado_bgr, (0, fila), (sobel8Umbralizado_bgr.shape[1]-1, fila), (0, 0, 255), 1)
for columna in columnas_superiores:
    cv2.line(sobel8Umbralizado_bgr, (columna, 0), (columna, sobel8Umbralizado_bgr.shape[0]-1), (255, 0, 0), 1)
plt.imshow(cv2.cvtColor(sobel8Umbralizado_bgr, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```

#### Resultado
<img src="Images/task2Result1.png" width="1200" height="400" alt="Resultado Task-2.1">
<img src="Images/task2Result2.png" width="800" height="400" alt="Resultado Task-2.2">
<img src="Images/task2Result3.png" width="400" height="400" alt="Resultado Task-2.3">


### Tarea 3

En este proyecto, desarrollé un demostrador que captura imágenes de la cámara web y aplica diferentes funciones de OpenCV para mostrar lo aprendido en las prácticas anteriores. El programa muestra la imagen original junto con tres efectos distintos en una sola ventana.

#### Captura de la Cámara Web:
   - Iniciamos la captura de video desde la cámara web y redimensionamos el frame para facilitar el procesamiento.

#### Efecto de Detección de Bordes (Esquina Superior Derecha):
   - Convertimos el frame a escala de grises y aplicamos un desenfoque Gaussiano para reducir el ruido.
   - Utilizamos el detector de bordes de Canny para resaltar los bordes en la imagen.
   - Convertimos la imagen resultante a BGR para poder combinarla con otros efectos.

   ```python
   # Efecto esquina superior derecha: Canny
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   blur = cv2.GaussianBlur(gray, (5, 5), 0)
   edges = cv2.Canny(blur, 50, 150)
   edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
   ```

#### Detección de Contraste (Esquina Inferior Derecha):
   - Encontramos los píxeles más brillantes y más oscuros en la imagen en escala de grises.
   - Marcamos estos píxeles en el frame original con círculos y etiquetas.

   ```python
   # Efecto esquina inferior derecha: Detector de contraste
   min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
   contrastDetector = frame.copy()
   cv2.circle(contrastDetector, min_loc, 10, (255, 0, 0), 2)  # Círculo azul
   cv2.putText(contrastDetector, 'Oscuro', (min_loc[0] + 10, min_loc[1]),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
   cv2.circle(contrastDetector, max_loc, 10, (0, 0, 255), 2)  # Círculo rojo
   cv2.putText(contrastDetector, 'Brillante', (max_loc[0] + 10, max_loc[1]),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
   ```

#### Efecto Pop Art:
   - Dividimos el frame en cuatro cuadrantes.
   - Aplicamos diferentes mapas de color a cada cuadrante para crear un efecto de arte pop.

   ```python
   # Efecto Pop Art
   height, width, _ = frame.shape
   pop_art = np.zeros_like(frame)
   pop_art[0:height//2, 0:width//2] = cv2.applyColorMap(frame[0:height//2, 0:width//2], cv2.COLORMAP_HSV)
   pop_art[0:height//2, width//2:width] = cv2.applyColorMap(frame[0:height//2, width//2:width], cv2.COLORMAP_OCEAN)
   pop_art[height//2:height, 0:width//2] = cv2.applyColorMap(frame[height//2:height, 0:width//2], cv2.COLORMAP_PINK)
   pop_art[height//2:height, width//2:width] = cv2.applyColorMap(frame[height//2:height, width//2:width], cv2.COLORMAP_SPRING)
   ```

#### Combinación y Visualización:
   - Combinamos la imagen original y los efectos en una sola ventana para mostrar los resultados en tiempo real.
   - Organizamos las imágenes de la siguiente manera:
     - **Arriba:** Imagen original y detección de bordes.
     - **Abajo:** Efecto Pop Art y detección de contraste.

   ```python
   # Combinamos las imágenes
   combined_top = np.hstack((frame, edges_colored))
   combined_bottom = np.hstack((pop_art, contrastDetector))
   combined = np.vstack((combined_top, combined_bottom))
   ```

#### Resultado
<img src="Images/task3Result.png" width="700" height="600" alt="Resultado Task-3">

### Tarea 4

En esta tarea, hemos desarrollado un programa que imita la instalación artística **"My Little Piece of Privacy"** de **Niklas Roy**. Utilizamos técnicas de procesamiento de video en tiempo real con **OpenCV** para crear una cortina digital animada que sigue el movimiento de una persona.

#### Descripción del Proyecto

El objetivo es simular una cortina que se mueve horizontalmente para cubrir la posición de una persona detectada en el video capturado por la cámara web. La cortina sigue el movimiento de la persona, ocultándola parcialmente y recreando el efecto de la instalación original.

#### Conceptos y Algoritmos Utilizados

- **Captura de Video en Tiempo Real**: Utilizamos la cámara web para obtener el flujo de video.

  ```python
  cap = cv2.VideoCapture(0)
  ```

- **Sustracción de Fondo (Background Subtraction)**: Aplicamos un algoritmo de sustracción de fondo para detectar movimiento en la escena y obtener una máscara de primer plano que resalta las áreas con cambio. Ajustamos los parámetros para mejorar la detección.

  ```python
  background_subtractor = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=200)
  fgmask = background_subtractor.apply(blurred_frame)
  ```

- **Procesamiento de Imágenes**: Convertimos el fotograma a escala de grises y aplicamos un desenfoque Gaussiano para reducir el ruido y mejorar la detección de movimiento.

  ```python
  frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  blurred_frame = cv2.GaussianBlur(frame_gray, (5, 5), 0)
  ```

- **Umbralización y Operaciones Morfológicas**: Realizamos una umbralización para binarizar la máscara de movimiento y aplicamos operaciones morfológicas para eliminar el ruido y pequeños objetos no deseados.

  ```python
  _, fgmask_thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  fgmask_clean = cv2.morphologyEx(fgmask_thresh, cv2.MORPH_OPEN, kernel)
  ```

- **Detección de Contornos**: Detectamos contornos en la máscara limpia para encontrar las áreas en movimiento. Seleccionamos el contorno más grande, asumiendo que es la persona.

  ```python
  contours, _ = cv2.findContours(fgmask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  largest_contour = max(contours, key=cv2.contourArea)
  x, y, w, h = cv2.boundingRect(largest_contour)
  ```

- **Posicionamiento y Animación de la Cortina**: Calculamos la posición objetivo de la cortina basada en la posición horizontal de la persona detectada. Utilizamos una interpolación suave para animar el movimiento de la cortina.

  ```python
  target_position = x
  curtain_position += (target_position - curtain_position) * animation_speed
  ```

- **Superposición de Imágenes con Transparencia**: Sobreponemos una imagen de una cortina sobre el fotograma original, utilizando el canal alpha para manejar la transparencia y lograr una integración natural.

  ```python
  frame[y_start:y_end, x_start:x_end] = blend_transparent(
      frame[y_start:y_end, x_start:x_end], curtain_resized
  )
  ```

  La función `blend_transparent` gestiona la mezcla de la cortina con el fondo teniendo en cuenta la transparencia.

- **Efecto Espejo**: Reflejamos horizontalmente el fotograma para que la interacción sea más intuitiva, similar a un espejo.

  ```python
  mirrored_frame = cv2.flip(frame, 1)
  ```

#### Visualización

El resultado es una ventana de video donde la cortina digital se mueve para cubrir a la persona detectada. La cortina sigue suavemente el movimiento, proporcionando una experiencia interactiva.

<img src="Images/task4Result.gif" width="700" alt="Resultado Tarea 4">

#### Conclusiones

Este proyecto demuestra cómo combinar técnicas de visión por computadora para recrear una obra de arte interactiva. La detección de movimiento, procesamiento de imágenes y animación se integran para lograr un efecto que responde en tiempo real a la presencia del usuario.

## Referencias y bibliografía

- OpenCV Documentation: https://docs.opencv.org/
- NumPy Documentation: https://numpy.org/doc/
- Matplotlib Documentation: https://matplotlib.org/stable/contents.html
- Niklas Roy - **"My Little Piece of Privacy"**: [niklasroy.com/project/100/my-little-piece-of-privacy](https://www.niklasroy.com/project/100/my-little-piece-of-privacy)
- OpenCV Documentation: [docs.opencv.org](https://docs.opencv.org/)
- NumPy Documentation: [numpy.org/doc](https://numpy.org/doc/)
- Tutoriales de Python y OpenCV: [pyimagesearch.com](https://www.pyimagesearch.com/)
