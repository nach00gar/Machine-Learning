# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:01:04 2022

@author: Ignacio Garach Vélez AA3

Fuentes de consulta utilizadas:

https://numpy.org/doc/stable/reference/generated/numpy.unique.html

https://numpy.org/doc/stable/reference/generated/numpy.unique.html

https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html

https://matplotlib.org/stable/gallery/mplot3d/subplot3d.html

https://matplotlib.org/stable/tutorials/colors/colormaps.html

https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html

https://www.freecodecamp.org/espanol/news/la-guia-definitiva-del-paquete-numpy-para-computacion-cientifica-en-python/


"""
"""Ejercicio 1. Scatter Plot"""

from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

bd = load_iris()                                #Cargamos la base de datos de flor Iris.

X = bd.data                                     #Tomamos por un lado las características en X y las clases correspondientes en y.
y = bd.target
classes = bd.target_names                       #Guardamos los nombres de cada clase.

X_firstthird = X[: , ::2]                       #Obtenemos las características primera y tercera, esto es, la columna 0 y la 2 (columnas desde 0 con paso 2).

colours = ['red', 'green', 'blue']              #Definimos los colores para las 3 clases.
groupcount = np.unique(y, return_counts=True)   #Unique devuelve, tanto las clases distintas en y, como la cantidad de cada uno de ellos(Opción return_counts).


for i in range(len(classes)):                   #Este bucle, pinta cada terna (x0, x2, y) de cada clase, con el color indicado en colours.
    if i==0:                                    #En el primer elemento, empieza desde el principio y acaba justo antes de la longitud de dicha clase.
        ini=0
        fin=groupcount[1][0]
    else:                                       #Para el resto, comienza en el índice suma de todos los elementos de las clases ya pintadas.
        ini=groupcount[1][0:i].sum()
        fin=groupcount[1][0:i+1].sum()
    plt.scatter(X_firstthird[ini:fin, [0]], X_firstthird[ini:fin, [1]], c=colours[i], label=classes[i])     #Utilizamos los ini y fin calculados para acceder a las características, colores y nombres de las clases para la leyenda.


plt.xlabel("Sepal Length (cm)")                 #Definimos el nombre de los ejes y activamos la leyenda ya definida.
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()



"""Ejercicio 2. Training & Test Split"""

pt = 0.8                                        #Definimos el porcentaje del conjunto de datos que servirá de entrenamiento.

combined = list(zip(X, y))                      #Combinamos cada vector de características con su clase, y así podemos hacer la reordenación aleatoria de los datos.
np.random.seed(13)                              #Marcamos la semilla, por si queremos repetir el experimento con la misma ordenación
np.random.shuffle(combined)

csetosa = 0                                     #Definimos contadores que, a posteriori, servirán para controlar la uniformidad de clases en Train/Test, en concreto llevarán la cuenta del número de elementos de cada especie en el conjunto de entrenamiento.
cversicolor = 0
cvirginica = 0

train = []
test = []

n = len(combined)                               #Cantidad de datos (Pares XY)
c = len(classes)                                #Cantidad de clases                      
npc = int(n / c)                                #Cantidad de datos por clase, sabemos que es uniforme por el ejercicio anterior, sino se podría generalizar (con unique y return_counts).

for (a, b) in combined:                         #Este bucle recorre los datos desordenados y va introduciendo un 80 por ciento en train para cada clase, si se llena ese cupo (40 en este caso), se envían a train.
    if b==0:
        if csetosa < npc*pt:
            train.append((a, b))
            csetosa = csetosa+1
        else:
            test.append((a, b))
            
    if b==1:
        if cversicolor < npc*pt:
            train.append((a, b))
            cversicolor = cversicolor+1
        else:
            test.append((a, b))
    if b==2:
        if cvirginica < npc*pt:
            train.append((a, b))
            cvirginica = cvirginica+1
        else:
            test.append((a, b))
        
train = np.array(train, dtype=object)
test = np.array(test, dtype=object)

print('--- Clase setosa ---\n')
print('Ejemplos train:  ' + str(np.where(train[:, 1]==0)[0].size) + '\n')       #Verificamos la cantidad de elementos en Train y Test para cada clase, hacemos uso de la claúsula condicional where.
print('Ejemplos test:  ' + str(np.where(test[:, 1]==0)[0].size) + '\n')

print('--- Clase versicolor ---\n')
print('Ejemplos train:  ' + str(np.where(train[:, 1]==1)[0].size) + '\n')
print('Ejemplos test:  ' + str(np.where(test[:, 1]==1)[0].size) + '\n')


print('--- Clase virginica ---\n')
print('Ejemplos train:  ' + str(np.where(train[:, 1]==2)[0].size) + '\n')
print('Ejemplos test:  ' + str(np.where(test[:, 1]==2)[0].size) + '\n')

print('Clase de los ejemplos de entrenamiento:\n')                              #Mostramos por consola las clases que resultan en cada conjunto
print(train[:, 1])
print('Clase de los ejemplos de test:\n')
print(test[:, 1])





"""Ejercicio 3. 2D Plots """

dominio = np.linspace(0, 4*np.math.pi, 100)                         #Definimos los puntos del dominio que tomaremos para aplicar las funciones. 100 puntos equiespaciados en (0, 4pi).

f1 = np.sinh(dominio) * (10**(-5))                                  #Aplicamos las funciones de forma "vectorial" sobre el dominio, se usa la aproximación de las mismas programada en Numpy .
f2 = np.cos(dominio)
f3 = np.tanh(2*np.sin(dominio) - 4*np.cos(dominio))

fig, ax = plt.subplots()                                            #Definimos la figura y las distintas gráficas, con el color indicado en el guión y con el estilo dashed correspondiente a la imagen del guión. También definimos la leyenda(label).
ax.plot(dominio, f1, color='green', linestyle='dashed', label='y = 1e-5*sinh(x)')
ax.plot(dominio, f2, color='black', linestyle='dashed', label='y = cos(x)')
ax.plot(dominio, f3, color='red', linestyle='dashed', label='y = tanh(2*sin(x)-4*cos(x))')
ax.legend()
plt.show()



"""Ejercicio 4. 3D Plots """

from matplotlib import cm                                           #Importamos los colormaps, para utilizar los mismos que en el guión, coolwarm y viridis.

figure = plt.figure(figsize=plt.figaspect(0.5))                     #Definimos la figura de forma que sea el doble de ancha que de alta, para que quepan ambas figuras.

domX1 = np.linspace(-6.0, 6.0, 100)                                 #Definimos los puntos equiespaciados para los ejes coordenados x e y.
domY1 = np.linspace(-6.0, 6.0, 100)
domX1, domY1 = np.meshgrid(domX1, domY1)                            #Esta función prepara los pares de puntos para evaluar, según le indicamos.

Z1 = 1 - np.abs(domX1 + domY1) - np.abs(domY1 - domX1)


domX = np.linspace(-2.0, 2.0, 100)                                  #Definimos los puntos equiespaciados para los ejes coordenados x e y.
domY = np.linspace(-2.0, 2.0, 100)
domX, domY = np.meshgrid(domX, domY)
Z2 = domX*domY*np.exp(-domX**2-domY**2)                             #Aplicamos la función de 2 variables del mismo modo que en el ejercicio anterior.

ax = figure.add_subplot(1, 2, 1, projection='3d')                   #Añadimos una subfigura para el primer gráfico.
surf1 = ax.plot_surface(domX1, domY1, Z1, cmap=cm.coolwarm)         #Dibujamos la primera superficie con el esquema de colores coolwarm.
ax.set_title("Pirámide")                                            #Definimos su título en su subfigura.

ax = figure.add_subplot(1, 2, 2, projection='3d')                   #Repetimos el proceso para la segunda gráfica, en este caso con el esquema viridis.
surf2 = ax.plot_surface(domX, domY, Z2, cmap=cm.viridis)            
ax.set_title(r'$x \dot y \dot e^{(-x^2-y^2)}$')                     #Definimos su título en su subfigura. En este caso se usa laTEX para las fórmulas. 

plt.show()


