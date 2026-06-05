import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rtree import index
import face_recognition

DATASET_PATH = "D:\\temp\\lfw"

coleccion = []
for path in glob.iglob(os.path.join(DATASET_PATH, "**", "*.jpg")):
    person = path.split("\\")[-2]
    coleccion.append({"person": person, "path": path})

coleccion = pd.DataFrame(coleccion)
coleccion.head(10)

def mostrarFotos(coleccion, posiciones):
    plt.figure(figsize=(16, 10))
    i = 0
    for idx in posiciones:
        img = plt.imread(coleccion.path.iloc[idx])
        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.title(coleccion.person.iloc[idx] + str(img.shape))
        plt.xticks([])
        plt.yticks([])
        i += 1
    plt.tight_layout()
    plt.show()

posiciones = list(range(0, 16))
mostrarFotos(coleccion, posiciones)