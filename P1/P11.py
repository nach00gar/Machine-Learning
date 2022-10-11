# -*- coding: utf-8 -*-
"""
TRABAJO 1.1 
Nombre Estudiante: Ignacio Garach Vélez

"""

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1.1 y 1.2\n')
#En primer lugar se definen la función y las derivadas
def E(u,v):
    return (u * v * np.exp((-u**2 - v**2)))**2   

#Derivada parcial de E con respecto a u
def dEu(u,v):
    return -2 * u * (2 * u**2 - 1) * v**2 * np.exp(-2*(u**2 + v**2))
    
#Derivada parcial de E con respecto a v 
def dEv(u,v):
    return -2 * v * (2 * v**2 - 1) * u**2 * np.exp(-2*(v**2 + u**2))

#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])


def gradient_descent(w_ini, lr, grad_fun, fun, epsilon, max_iters):
    # gradiente descendente
    '''Entrada:
            w_ini: Punto de inicio
            lr: Tasa de aprendizaje(learning rate)
            grad_fun: Gradiente de la función a minimizar
            fun: Función a minimizar(error)
            epsilon: Error Admisible
            max_iters: Número máximo de iteraciones
            
        Salida:
            w: Punto donde se alcanza el mínimo (Vector de pesos del modelo entrenado)
            iterations: Número de iteraciones realizado'''
    iterations = 0
    w = w_ini.copy()    
    error_actual = fun(w[0], w[1])                                  #Calculamos el error inicial
    while max_iters > iterations and error_actual > epsilon:        #Comenzamos a iterar
        w = w - (lr * grad_fun(w[0], w[1]))                         #Desciendo en la dirección del gradiente en proporción a la tasa de aprendizaje
        error_actual = fun(w[0], w[1])                              #Recalculo el error
        iterations = iterations + 1
    return w, iterations

def gradient_descent_extrareturns(w_ini, lr, grad_fun, fun, epsilon, max_iters):
    #Es exactamente el mismo algoritmo anterior, pero con nuevas salidas útiles para graficar su funcionamiento
    '''Entrada:
            w_ini: Punto de inicio
            lr: Tasa de aprendizaje(learning rate)
            grad_fun: Gradiente de la función a minimizar
            fun: Función a minimizar(error)
            epsilon: Error Admisible
            max_iters: Número máximo de iteraciones
            
        Salida:
            w: Punto donde se alcanza el mínimo (Vector de pesos del modelo entrenado)
            iterations: Número de iteraciones realizado
            traza: Vector con los puntos que va obteniendo el algoritmo
            evol: Vector con los errores(valor de la función) que va obteniendo'''
    iterations = 0
    w = w_ini.copy()
    error_actual = fun(w[0], w[1])
    traza = [w_ini]
    evol = [error_actual]
    while max_iters > iterations and error_actual > epsilon:
        
        w = w - (lr * grad_fun(w[0], w[1]))
        
        error_actual = fun(w[0], w[1])
        iterations = iterations + 1
        traza.append(w)
        evol.append(error_actual)
    return w, iterations, traza,  evol  





'''
Esta función muestra una figura 3D con la función a optimizar junto con el 
óptimo encontrado y la ruta seguida durante la optimización. Esta función, al igual
que las otras incluidas en este documento, sirven solamente como referencia y
apoyo a los estudiantes. No es obligatorio emplearlas, y pueden ser modificadas
como se prefiera. 
    rng_val: rango de valores a muestrear en np.linspace()
    fun: función a optimizar y mostrar
    ws: conjunto de pesos (pares de valores [x,y] que va recorriendo el optimizador
                           en su búsqueda iterativa del óptimo)
    colormap: mapa de color empleado en la visualización
    title_fig: título superior de la figura
    
Ejemplo de uso: display_figure(2, E, ws, 'plasma','Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
'''
def display_figure(rng_val, fun, ws, colormap, title_fig, labels):
    # https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
    from mpl_toolkits.mplot3d import Axes3D
    x = np.linspace(-rng_val, rng_val, 100)
    y = np.linspace(-rng_val, rng_val, 100)
    X, Y = np.meshgrid(x, y)
    Z = fun(X, Y) 
    fig = plt.figure()
    ax = Axes3D(fig,auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                            cstride=1, cmap=colormap, alpha=.6)
    if len(ws)>0:
        ws = np.asarray(ws)
        min_point = np.array([ws[-1,0],ws[-1,1]])
        min_point_ = min_point[:, np.newaxis]
        ax.plot(ws[:-1,0], ws[:-1,1], E(ws[:-1,0], ws[:-1,1]), 'r*', markersize=5)
        ax.plot(min_point_[0], min_point_[1], E(min_point_[0], min_point_[1]), 'r*', markersize=10)
    if len(title_fig)>0:
        fig.suptitle(title_fig, fontsize=16)
    ax.set_xlabel(labels[0])    #Se ha modificado para poder pasarle como parámetro las etiquetas de los ejes
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
 
#Defino los parámetros que recibe el algoritmo y lo aplico en el ejercicio 1.2
    
eta = 0.1 
maxIter = 10000000000
error2get = 1e-8
initial_point = np.array([0.5,-0.5])

w, it, traza, evol = gradient_descent_extrareturns(initial_point, eta, gradE, E, error2get, maxIter)


print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
labels = ['u', 'v', 'E(u, v)'] #Vector con las etiquetas de los ejes en la gráfica
display_figure(2.1, E, traza, 'plasma', 'Ejercicio 1.2', labels) #Graficamos con la función suministrada el camino seguido por el algoritmo
plt.show()


#Vamos a reflejar en una gráfica como va disminuyendo el error o valor de la función conforme iteramos

fig, ax = plt.subplots()
iteraciones = np.arange(it+1)
plt.plot(iteraciones+1, evol)
plt.xlabel('Número de iteraciones')
plt.ylabel('Valor de E')
plt.title(r'Learning rate $\eta = 0.1$')
plt.show()

#Probamos a modificar la tasa de aprendizaje, se comenta en la memoria el resultado.

eta = 0.5
maxIter = 10000000000
error2get = 1e-8
initial_point = np.array([0.5,-0.5])

w, it, traza, evol = gradient_descent_extrareturns(initial_point, eta, gradE, E, error2get, maxIter)

print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
labels = ['u', 'v', 'E(u, v)']
display_figure(2.1, E, traza, 'plasma', 'Ejercicio 1.2', labels)
plt.show()

#De nuevo mostramos la gráfica con los valores conforme iteramos, serán claves en la comparación

fig, ax = plt.subplots()
iteraciones = np.arange(it+1)
plt.plot(iteraciones+1, evol)
plt.xlabel('Número de iteraciones')
plt.ylabel('Valor de E')
plt.title(r'Learning rate $\eta = 0.5$')
plt.xscale('log')
plt.show()





print('Ejercicio 1.3\n')

#En primer lugar se definen la función y las derivadas de la f

def f(x, y):
    return x**2 + 2 * y**2 + 2 * np.sin(2 * np.pi * x) * np.sin(np.pi * y)     

#Derivada parcial de f con respecto a x
def dfx(x,y):
    return 2 * x + 4 * np.pi * np.sin(np.pi * y) * np.cos(2 * np.pi * x)
    
#Derivada parcial de f con respecto a y 
def dfy(x,y):
    return 4 * y + 2 * np.pi * np.sin(2 * np.pi * x) * np.cos(np.pi * y)

#Gradiente de f
def gradf(x,y):
    return np.array([dfx(x,y), dfy(x,y)])

def display_figure(rng_val, fun, ws, colormap, title_fig, labels):
    # https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
    from mpl_toolkits.mplot3d import Axes3D
    x = np.linspace(-rng_val, rng_val, 100)
    y = np.linspace(-rng_val, rng_val, 100)
    X, Y = np.meshgrid(x, y)
    Z = fun(X, Y) 
    fig = plt.figure()
    ax = Axes3D(fig,auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                            cstride=1, cmap=colormap, alpha=.6)
    if len(ws)>0:
        ws = np.asarray(ws)
        min_point = np.array([ws[-1,0],ws[-1,1]])
        min_point_ = min_point[:, np.newaxis]
        ax.plot(ws[:-1,0], ws[:-1,1], f(ws[:-1,0], ws[:-1,1]), 'r*', markersize=5)
        ax.plot(min_point_[0], min_point_[1], f(min_point_[0], min_point_[1]), 'r*', markersize=10)
    if len(title_fig)>0:
        fig.suptitle(title_fig, fontsize=16)
    ax.set_xlabel(labels[0])    #Se ha modificado para poder pasarle como parámetro las etiquetas de los ejes
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

#Lanzamos el algoritmo sobre la función con los parámetros del enunciado

eta = 0.01 
maxIter = 50
error2get = -np.Infinity        #Como en este apartado queremos hacer siempre las 50 iteraciones, eliminos el otro criterio de parada
initial_point = np.array([-1.0, 1.0])
w, it, traza, evol = gradient_descent_extrareturns(initial_point, eta, gradf, f, error2get, maxIter)


print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

#Vamos a generar las mismas gráficas que en el otro ejercicio con el mismo proceder

labels = ['x', 'y', 'f(x, y)']
display_figure(5, f, traza, 'jet', 'Ejercicio 1.3', labels)
plt.show()



iteraciones = np.arange(it+1)
plt.clf()
plt.plot(iteraciones, evol)
plt.xlabel('Número de iteraciones')
plt.ylabel('Valor de f')
plt.title(r'Learning rate $\eta = 0.01$')
plt.show()

#Cambiamos la tasa de aprendizaje

eta = 0.1 
maxIter = 50
error2get = -np.Infinity
initial_point = np.array([-1.0, 1.0])
w, it, traza, evol = gradient_descent_extrareturns(initial_point, eta, gradf, f, error2get, maxIter)

iteraciones = np.arange(it+1)
plt.clf()
plt.plot(iteraciones, evol)
plt.xlabel('Número de iteraciones')
plt.ylabel('Valor de f')
plt.title(r'Learning rate $\eta = 0.1$')
plt.show()



display_figure(5, f, traza, 'jet', 'Ejercicio 1.3', labels)
plt.show()

print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')



#Definimos las tasas y puntos de partida que se nos piden
error2get = -np.Infinity
maxIter = 50
distintosinicios = [np.array([-0.5, -0.5]), np.array([1.0, 1.0]), np.array([2.1, -2.1]), np.array([-3.0, 3.0]), np.array([-2.0, 2.0])]
tasas = [0.01, 0.1]

#Calculamos los resultados, tanto mínimos como valores donde se alcanzan para todas las opciones que se piden
for ini in distintosinicios:
    for t in tasas:
        w, it, traza, evol = gradient_descent_extrareturns(ini, t, gradf, f, error2get, maxIter)
        print ('Resultados para punto inicial ', ini, ' y η = ', t, ' : ')
        print ('Numero de iteraciones: ', it)
        print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
        print ('Valor de la función en el mínimo: ' + str(f(w[0], w[1])))
        
        
