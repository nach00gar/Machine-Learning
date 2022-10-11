# -*- coding: utf-8 -*-
"""
TRABAJO 1.2
Nombre Estudiante: Ignacio Garach Vélez
Fuentes: 
    
https://numpy.org/doc/stable/reference/generated/numpy.hstack.html

https://www.tutorialspoint.com/matplotlib/matplotlib_contour_plot.htm

https://numython.github.io/posts/2016/02/graficas-de-contorno-en-matplotlib/
"""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)
print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 2.1\n')

label5 = 1
label1 = -1
# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

#Definamos en primer lugar los algoritmos y las funciones que usarán

# Funcion para calcular el error cuadrático medio
def Err(x,y,w):
    '''Entrada:
            x: Matriz de datos, tiene tantas filas como datos y columnas como características tiene cada dato
            y: Vector con las clases correspondientes a los datos x
            w: Vector de pesos para el cual se quiere calcular el error
            
        Salida:
            Error cuadrático medio'''
    return (np.linalg.norm(x.dot(w) - y)**2)/len(x)

# Función que calcula el gradiente del error

def gradError(x,y,w):
    '''Entrada:
            x: Matriz de datos, tiene tantas filas como datos y columnas como características. Se utiliza con un subconjunto de datos distinto en SGD cada vez.
            y: Vector con las clases correspondientes a los datos x
            w: Vector de pesos para el cual se quiere calcular el gradiente
            
        Salida:
            Derivada del error cuadrático medio'''
    return 2*(x.T.dot(x.dot(w) - y))/x.shape[0]


# Gradiente Descendente Estocastico
def sgd(x, y, lr, max_iters, minibatchlen, epsilon):
    '''Entrada:
            x: Matriz de datos, tiene tantas filas como datos y columnas como características.
            y: Vector con las clases correspondientes a los datos x
            lr: Tasa de aprendizaje(learning rate)
            max_iters: Número de iteraciones a realizar
            minibatchlen: Tamaño de los minibatches
            epsilon: Error admisible para terminar antes
            
        Salida:
            Vector de pesos para el problema de regresión'''
    w = np.zeros(x.shape[1])                                                        #Inicializamos a 0 el vector de pesos para comenzar a iterar
    
    random_order = np.random.permutation(np.arange(x.shape[0]))                     #Realizamos una permutación aleatoria de tamaño el número de datos. Servirá para tomar despues conjuntos aleatorios de datos de tamaño minibatchlen.
    inicio = 0                                                                      #Inicializo la posición desde donde empezar a coger índices
    i = 0
    error = Err(x, y, w)                                                            #Calculamos el error en partida
    while i < max_iters and error > epsilon:
        if inicio + minibatchlen < x.shape[0]:                                      #Mientras no agotemos todos los datos, vamos tomando los índices en grupos del tamaño del minibatch
            minibatchpositions = random_order[inicio : inicio + minibatchlen]
        else:                                                                       #Si no hay datos suficientes para completar un minibatch, utilizamos los que queden.
            minibatchpositions = random_order[inicio : x.shape[0]]
        inicio = inicio + minibatchlen                                              #Actualizamos el índice de inicio para el siguiente mini-batch
        w = w - lr*gradError(x[minibatchpositions, :], y[minibatchpositions], w)    #Aplicamos el descenso de gradiente pero calculando el gradiente sólo con el minibatch considerado
        error = Err(x, y, w)                                                        #Actualizo el error 
        if inicio > len(x):                                                         #Si he agotado los datos reinicio la permutación y el índice de inicio
            random_order = np.random.permutation(np.arange(x.shape[0]))
            inicio = 0
            i+=1                                                                    #Consideramos una iteración como un recorrido completo de los datos, por tanto la incrementamos
    return w

#Pseudoinversa utilizando la descomposición en valores singulares 
def pseudoinverse(x, y):
    '''Entrada:
            x: Matriz de datos, tiene tantas filas como datos y columnas como características.
            y: Vector con las clases correspondientes a los datos x'''
    u, d, v = np.linalg.svd(x)
    return v.transpose() @ np.diag(d**(-2)) @ v @ x.transpose() @ y


#Una vez implementados los algoritmos pasamos a aplicarlos sobre los datos


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

labels = [label5, label1]
colours = {label5: 'red', label1: 'green'}
values = {label5: '5', label1: '1'}
plt.clf()

#Representamos en un scatter plot las diferentes clases
for l in labels:
    index = np.where(y == l)
    plt.scatter(x[index, 1], x[index, 2], c=colours[l], label=values[l])

plt.xlabel('Valor de gris')
plt.ylabel('Simetría')
w = pseudoinverse(x, y)                                                             #Llamamos al algoritmo exacto de pseudoinversa
print ('Bondad del resultado para pseudoinversa:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))                                            #Para calcular el error fuera de la muestra, utilizamos los datos de test

def sign(x):
	if x >= 0:
		return 1
	return -1

#Calculamos el accuracy en la muestra y en el test

prediccion = np.array([w @ dato for dato in x ]) #Obtenemos las distancias al plano, nos interesa solo el signo para la predicción
accuracy = np.count_nonzero(np.array([sign(p)==r for (p, r) in zip(prediccion, y)]))/len(y) #Comprobamos la predicción con los valores reales, contamos en cuantas coinciden y dividimos entre la cantidad de datos
print("Exactitud en la muestra (accuracy): ", accuracy)

prediccion = np.array([w @ dato for dato in x_test ]) #Obtenemos las distancias al plano, nos interesa solo el signo para la predicción
accuracy = np.count_nonzero(np.array([sign(p)==r for (p, r) in zip(prediccion, y_test)]))/len(y_test) #Comprobamos la predicción con los valores reales, contamos en cuantas coinciden y dividimos entre la cantidad de datos
print("Exactitud fuera de la muestra (accuracy): ", accuracy)

dom = np.linspace(0, 0.5, 10)
recta = (-w[0] - w[1]*dom)/w[2]    #La recta que se muestra es el resultado de proyectar al plano de los ejes coordenados (x1, x2)     

plt.plot(dom, recta, label='Regresión lineal mediante Pseudoinversa')
plt.legend()
plt.show()


#Ahora lanzamos el algoritmo de gradiente estocástico descendente sobre los mismos datos
eta = 0.01
max_iters = 100 
bsize = 32
error = 1e-8


w = sgd(x, y, eta, max_iters, bsize, error)
print(w)
rectasgd = (-w[0] - w[1]*dom)/w[2]
plt.plot(dom, rectasgd, label='Regresión lineal mediante SGD')
plt.legend()

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

#Calculamos el accuracy en la muestra y en el test

prediccion = np.array([w @ dato for dato in x ]) 
accuracy = np.count_nonzero(np.array([sign(p)==r for (p, r) in zip(prediccion, y)]))/len(y) 
print("Exactitud en la muestra (accuracy): ", accuracy)

prediccion = np.array([w @ dato for dato in x_test ]) 
accuracy = np.count_nonzero(np.array([sign(p)==r for (p, r) in zip(prediccion, y_test)]))/len(y_test) 
print("Exactitud fuera de la muestra (accuracy): ", accuracy)

input("\n--- Pulsar tecla para continuar --- AVISO: EL EXPERIMENTO LLEVA ALGO MÁS DE UN MINUTO\n")




print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

def sign(x):
	if x >= 0:
		return 1
	return -1

def f(x1, x2):
	return sign((x1-0.2)**2+x2**2-0.6) 

#Vamos a generar el conjunto de 1000 datos en el cuadrado de lados [-1, 1]

N = 1000
s = 1
ruido = 0.1                                                                         #Porcentaje ruidoso que se cambia en el apartado b

uniformes = simula_unif(N, 2, s)

plt.scatter(uniformes[:, 0], uniformes[:, 1], c='blue')                             #Pintamos el conjunto de puntos
plt.title('Puntos uniformemente muestreados N=1000')

clases = np.array([f(x[0], x[1]) for x in uniformes])                               #Calculamos la clase de cada punto usando la función f

pos = np.arange(len(clases))                                                        #Permutamos los enteros que son índices de del número de datos
np.random.shuffle(pos)                                                              
for i in range(int(ruido*len(clases))):                                             #Alteramos el signo en el ruido*100 por ciento de los datos
    clases[pos[i]] *= -1

#Representamos ahora las 2 clases generadas

labels = [1, -1]
colours = {1: 'red', -1: 'blue'}
plt.clf()
for l in labels:
    index = np.where(clases == l)
    plt.scatter(uniformes[index, 0], uniformes[index, 1], c=colours[l], label=l)
plt.legend()

eta = 0.01
max_iters = 100
bsize = 32
error = 1e-8

#Lanzamos el algoritmo para los datos generados.

uniformes = np.hstack((np.ones((len(clases), 1)), uniformes))                       #Añadir la columna de 1 para el término afín
w = sgd(uniformes, clases, eta, max_iters, bsize, error)
dom = np.linspace(-0.25, 0.25, 10)                                                  # Graficamos la función de nuevo sobre los datos
rectasgd = (-w[0] - w[1]*dom)/w[2]
plt.plot(dom, rectasgd, "g", label='SGD')
plt.legend(loc='upper right')
plt.xlim(-1, 1)
plt.ylim(-1, 1)


print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(uniformes, clases,w))

prediccion = np.array([w @ dato for dato in uniformes ]) #Obtenemos las distancias al plano, nos interesa solo el signo para la predicción
accuracy = np.count_nonzero(np.array([sign(p)==r for (p, r) in zip(prediccion, clases)]))/len(clases) #Comprobamos la predicción con los valores reales, contamos en cuantas coinciden y dividimos entre la cantidad de datos
print("Exactitud en la muestra (accuracy): ", accuracy)



#Ejercicio 2d

def generaMuestraNueva(pintar=False):
    '''Entrada:
            pintar: Booleano que indica si se desea que se dibujen las 2 clases. Por defecto no se hace.
        Salida:
                uniformes: Puntos(x)
                clases: Clases(f(x))'''
    uniformes = simula_unif(N, 2, s)                                                #Generamos los puntos
    clases = np.array([f(x[0], x[1]) for x in uniformes])                           #Calculamos su clase
    pos = np.arange(len(clases))
    np.random.shuffle(pos)
    for i in range(int(ruido*len(clases))):
        clases[pos[i]] *= -1
        
    if pintar:
        labels = [1, -1]
        colours = {1: 'red', -1: 'blue'}
        for l in labels:
            index = np.where(clases == l)
            plt.scatter(uniformes[index, 0], uniformes[index, 1], c=colours[l], label=l)
        plt.legend()
        
    uniformes = np.hstack((np.ones((len(clases), 1)), uniformes))                   #Añadimos una columna de 1`s como término afín para ser consistentes con la definición del algoritmo
    
    return uniformes, clases


#Preparamos los parámetros para el experimento

eta = 0.01
max_iters = 100
bsize = 32
error = 1e-8


experimento = 1000
ein = 0
eout = 0
ain = 0
aout = 0
for i in range(experimento):
    #print(i)
    x_train, y_train = generaMuestraNueva()                                         #Generamos los datos de entrenamiento
    w = sgd(x_train, y_train, eta, max_iters, bsize, error)                         #Aplicamos el algoritmo. Entrenamos el modelo.
    x_test, y_test = generaMuestraNueva()                                           #Generamos los datos de test
    ein += Err(x_train, y_train, w)                                                 #Calculamos el error dentro de la muestra
    eout += Err(x_test, y_test, w)                                                  #Calculamos el error dentro de la muestra
    #Dejo comentados estos cálculos de exactitud para acelerar las ejecuciones
    #prediccion = np.array([w @ dato for dato in x_train ])
    #ain += np.count_nonzero(np.array([sign(p)==r for (p, r) in zip(prediccion, y_train)]))/len(y_train) 
    #prediccion = np.array([w @ dato for dato in x_test ])
    #aout += np.count_nonzero(np.array([sign(p)==r for (p, r) in zip(prediccion, y_test)]))/len(y_test)

ein /= experimento
eout /= experimento
#ain /= experimento
#aout /= experimento
print ('\nBondad promedio para grad. descendente estocastico en 1000 experimentos:\n')
print ("Ein: ", ein)
print ("Eout: ", eout)
#print("Exactitud promedio en la muestra (accuracy): ", ain)
#print("Exactitud promedio fuera deen  la muestra (accuracy): ", aout)




input("\n--- Pulsar tecla para continuar --- AVISO: EL EXPERIMENTO LLEVA ALGO MÁS DE UN MINUTO\n")
#Ejercicio 2e
plt.clf()


def generaMuestraYNoLineales(pintar=False):
    '''Entrada:
            pintar: Booleano que indica si se desea que se dibujen las 2 clases. Por defecto no se hace.
        Salida:
                uniformes: Puntos con nuevas características hasta los términos cuadráticos (1, x1, x2, x1x2, x1^2, x2^2)
                clases: Clases(f(x))'''
    x, y = generaMuestraNueva(pintar)
    x1x2 = np.array(x[:, 1]*x[:, 2]).reshape(len(clases), 1)                        #Calculo las nuevas características y las pongo en forma de columna
    x1cuadrado = np.array(x[:, 1]**2).reshape(len(clases), 1)
    x2cuadrado = np.array(x[:, 2]**2).reshape(len(clases), 1)
    x = np.hstack((x, x1x2, x1cuadrado, x2cuadrado))                                #Las uno a los datos originales
    
    return x, y


def evaluarPolinomio(x1, x2, w):
    '''Entrada:
            x1: Supuesto valor de la característica x1
            x2: Supuesto valor de la característica x2
            w: Vector de pesos ya ajustado, son los coeficientes del polinomio
        Salida:
                Valor de evaluación del polinomio cuadrático, se utiliza para graficar la proyección de la superficie'''
    return (w[0] + w[1]*x1 + w[2]*x2 + w[3]*x1*x2 + w[4]*x1*x1 + w[5]*x2*x2)


#Aplicamos las funciones al ejercicio en cuestión

x, y = generaMuestraYNoLineales(True)

eta = 0.01
max_iters = 100
bsize = 32
error = 1e-8

w = sgd(x, y, eta, max_iters, bsize, error)

dom = np.linspace(-1, 1, 100)
domX, domY = np.meshgrid(dom, dom)
Z = evaluarPolinomio(domX, domY, w)
plt.contour(domX, domY, Z, 0)                   #Esta función se utiliza para mostrar la superficie tridimensional de regresión en el plano bidimensional coordenado x1,x2 


prediccion = np.array([w @ dato for dato in x ]) 
accuracy = np.count_nonzero(np.array([sign(p)==r for (p, r) in zip(prediccion, y)]))/len(y) 


#Mostramos el error asociado a la regresión
print ('Bondad del resultado para grad. descendente estocastico con términos cudráticos:\n')
print ("Ein: ", Err(x, y,w))
print("Exactitud en la muestra (accuracy): ", accuracy)



#Repetimos el experimento pero esta vez con los nuevos términos

experimento = 1000
ein = 0
eout = 0
ain = 0
aout = 0
for i in range(experimento):
    x_train, y_train = generaMuestraYNoLineales()
    w = sgd(x_train, y_train, eta, max_iters, bsize, error)
    x_test, y_test = generaMuestraYNoLineales()    
    ein += Err(x_train, y_train, w)
    eout += Err(x_test, y_test, w)
    #Dejo comentados estos cálculos de exactitud para acelerar las ejecuciones
    #prediccion = np.array([w @ dato for dato in x_train ])
    #ain += np.count_nonzero(np.array([sign(p)==r for (p, r) in zip(prediccion, y_train)]))/len(y_train) 
    #prediccion = np.array([w @ dato for dato in x_test ])
    #aout += np.count_nonzero(np.array([sign(p)==r for (p, r) in zip(prediccion, y_test)]))/len(y_test)

ein /= experimento
eout /= experimento
#ain /= experimento
#aout /= experimento

print ('\nBondad promedio para grad. descendente estocastico para polinomios cuadráticos en 1000 experimentos:\n')
print ("Ein: ", ein)
print ("Eout: ", eout)
#print("Exactitud promedio en la muestra (accuracy): ", ain)
#print("Exactitud promedio fuera deen  la muestra (accuracy): ", aout)


