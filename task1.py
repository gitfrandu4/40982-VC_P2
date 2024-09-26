import cv2  
import numpy as np
import matplotlib.pyplot as plt

#Lee imagen de archivo
img = cv2.imread('mandril.jpg') 

#Conversión de la imagen a niveles de grises de la imagen original en BGR
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Obtiene contornos con el operador de Canny
canny = cv2.Canny(gris, 250, 100)

col_counts = cv2.reduce(canny, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
row_counts = np.transpose(cv2.reduce(canny, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)) 

# Note: 
# - Transpose: changes the dimensions of the array, so the shape of the array is swapped
# - Transpose is used to get the row counts from the column counts

# Normalize by the number of rows and the maximum pixel value (255)
cols = col_counts[0] / (255 * canny.shape[0]) # % of white pixels per column
rows = row_counts[0] / (255 * canny.shape[1]) # % of white pixels per row

print(col_counts.shape)  # Should output (1, 512)
print(row_counts.shape)  # Should output (512, 1)

#Muestra dicha cuenta gráficamente
plt.figure()

plt.subplot(2, 2, 1)
plt.axis("off")
plt.title("Canny")
plt.imshow(canny, cmap='gray') 

plt.subplot(2, 2, 2)
plt.axis("off")
plt.title("Canny")
plt.imshow(np.rot90(canny), cmap='gray') 

plt.subplot(2, 2, 3)
plt.title("Respuesta de Canny")
plt.xlabel("Columnas")
plt.ylabel("% píxeles")
plt.plot(cols)

plt.subplot(2, 2, 4)
plt.title("Respuesta de Canny")
plt.xlabel("Filas")
plt.ylabel("% píxeles")
plt.plot(rows)

#Rango en x definido por las filas
plt.xlim([0, canny.shape[1]])
# plt.ylim([0, canny.shape[0]])

plt.show()
