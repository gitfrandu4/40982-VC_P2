# Práctica 2: Funciones básicas de OpenCV

Este repositorio contiene la **Práctica 2** sobre el uso de **OpenCV** y **NumPy** para el procesamiento de imágenes, donde se exploran tareas como la conversión a escala de grises, la detección de bordes y el análisis de píxeles en imágenes.

## Librerías utilizadas

[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy)](https://numpy.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=matplotlib)](https://matplotlib.org/)

## Autores

Este proyecto fue desarrollado por:

- [![GitHub](https://img.shields.io/badge/GitHub-Francisco%20Javier%20L%C3%B3pez%E2%80%93Dufour%20Morales-yellow?style=flat-square&logo=github)](https://github.com/gitfrandu4)
- [![GitHub](https://img.shields.io/badge/GitHub-Marcos%20V%C3%A1zquez%20Tascon-red?style=flat-square&logo=github)](https://github.com/DerKom)

## Tareas

### Tarea 1


### Tarea 2

```python
#Para poder ejecutar el código, unicamente ejecutando este bloque, vamos a incorporar elementos de los anteriores bloques
import cv2  
import numpy as np
import matplotlib.pyplot as plt

#Lee imagen de archivo
img = cv2.imread('mandril.jpg') 

#Si hay lectura correcta
if img is None:
    print('Imagen no encontrada')

#Conversión de la imagen a niveles de grises de la imagen original en BGR
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Gaussiana para suavizar la imagen original, eliminando altas frecuencias
ggris = cv2.GaussianBlur(gris, (3, 3), 0)

#Calcula en ambas direcciones (horizontal y vertical)
sobelx = cv2.Sobel(ggris, cv2.CV_64F, 1, 0)  # x
sobely = cv2.Sobel(ggris, cv2.CV_64F, 0, 1)  # y

#Combina ambos resultados
sobel = cv2.add(sobelx, sobely)

# Conversión a byte con openCV
sobel8 = cv2.convertScaleAbs(sobel)

# Realizamos un umbralizado a la imagen
_, sobel8Umbralizado = cv2.threshold(gris, 130, 255, cv2.THRESH_BINARY)

#Muestra un primer resultado, con la comparación de la imagen original, la que tiene aplicada el sobel y la que además tiene el umbralizado.
plt.figure(figsize=(12, 4))
plt.title("Imagen Clásica (gris)")
plt.axis("off")
plt.subplots_adjust(top=1)#Añadimos un espacio extra entre el título y los subplots

plt.subplot(1, 3, 1)
plt.axis("off")
plt.title('Original')
plt.imshow(gris, cmap='gray')

plt.subplot(1, 3, 2)
plt.axis("off")
plt.title('Sobel')
plt.imshow(sobel8, cmap='gray')

plt.subplot(1, 3, 3)
plt.axis("off")
plt.title('Sobel umbralizado')
plt.imshow(sobel8Umbralizado, cmap='gray')

plt.show()

col_counts = cv2.reduce(sobel8Umbralizado, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
#Normaliza en base al número de filas, primer valor devuelto por shape, y al valor máximo del píxel (255)
#El resultado será el número de píxeles blancos por columna
fil_counts = cv2.reduce(sobel8Umbralizado, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
fil_counts = fil_counts.flatten() #El resultado es una matriz de [num filas][1] y no intera tenerlo como un array undimensional, por así decirlo no tener un segunda array con solo el valor de la suma

cols = col_counts[0] / (255 * sobel8Umbralizado.shape[0])
filas = fil_counts / (255 * sobel8Umbralizado.shape[1])


#Dibujamos los gráficos que muestra el % de pixels blancos, tanto por columnas como por filas
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("Pixels Blancos por Columna")
plt.xlabel("Columnas")
plt.ylabel("% Pixels Blancos")
plt.plot(cols, color='blue')
plt.xlim([0, sobel8Umbralizado.shape[1]])
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title("Pixels Blancos por Fila")
plt.xlabel("Filas")
plt.ylabel("% Pixels Blancos")
plt.plot(filas, color='green')
plt.xlim([0, sobel8Umbralizado.shape[0]])
plt.grid(True)

plt.tight_layout(pad=3.0)
plt.show()

#Determinamos el valor de la fila y de la columna con el mayor cantidad de pixels blancos
maxfil = np.max(filas)
maxcol = np.max(cols)

#Ahora marcamos los límites (95%+) a partir de las cuales buscaremos la filas que cumplan dichos requisitos
limiteFila = maxfil * 0.95
limiteColumna = maxcol * 0.95

#Buscamos las filas y columnas cuyo % de pixels blancos es un 95% o mas con respecto a la fila con mayor cantidad de estos.
filasSuperioresAlLimite = np.where(filas >= limiteFila)
columnasSuperioresAlLimite = np.where(cols >= limiteColumna)

# Convertimos la imagen en escala de grises a BGR para poder dibujar en color
#gris_bgr = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
sobel8Umbralizado_bgr = cv2.cvtColor(sobel8Umbralizado, cv2.COLOR_GRAY2BGR)

# Remarcamos las filas que cumplen con el criterio
for fila in filasSuperioresAlLimite[0]:
    cv2.line(sobel8Umbralizado_bgr, (0, fila), (gris_bgr.shape[1]-1, fila), (0, 0, 255), 1)

# Remarcamos las columnas que cumplen con el criterio
for columna in columnasSuperioresAlLimite[0]:
    cv2.line(sobel8Umbralizado_bgr, (columna, 0), (columna, gris_bgr.shape[0]-1), (255, 0, 0), 1)

# Mostramos la imagen con las filas y columnas remarcadas
plt.figure(figsize=(4, 4))
plt.imshow(cv2.cvtColor(sobel8Umbralizado_bgr, cv2.COLOR_BGR2RGB))
plt.title('Imagen con filas y columnas remarcadas')
plt.axis('off')
plt.show()
```

### Tarea 3


### Tarea 4


## Referencias y bibliografía

- OpenCV Documentation: https://docs.opencv.org/
- NumPy Documentation: https://numpy.org/doc/
- Matplotlib Documentation: https://matplotlib.org/stable/contents.html