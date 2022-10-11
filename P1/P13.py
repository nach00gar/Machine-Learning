# -*- coding: utf-8 -*-
"""
TRABAJO 1.3
Nombre Estudiante: Ignacio Garach Vélez

@author: ignan
"""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)
print('EJERCICIO BONUS : MÉTODO DE NEWTON\n')

#Copiamos las definiciones de la función f y sus derivadas

#La diferencia es que para este método se utiliza información asociada a curvatura (segunda derivada)
#Por tanto introducimos también las derivadas parciales de segundo orden y la matriz hessiana


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

#Derivada parcial segunda de f con respecto a x
def d2fx(x,y):
    return 2 - 8 * np.pi**2 * np.sin(np.pi * y) * np.sin(2 * np.pi * x)

#Derivada parcial segunda de f con respecto a y
def d2fy(x,y):
    return 4 - 2 * np.pi**2 * np.sin(np.pi * y) * np.sin(2 * np.pi * x)

#Derivada parcial segunda de f con respecto a x e y (Derivadas cruzadas que coinciden)
def d2fxy(x,y):
    return 4 * np.pi**2 * np.cos(np.pi * y) * np.cos(2 * np.pi * x)

#Matriz Hessiana con las derivadas parciales de segundo orden
def hessf(x, y):
    return np.array([d2fx(x, y), d2fxy(x, y), d2fxy(x, y), d2fy(x, y)]).reshape((2, 2))  #Hacemos reshape para ponerle la forma matricial habitual del hessiano.

def relaxedNewtonRaphson(w_ini, lr, hess, grad_fun, fun, epsilon, max_iters):
    '''Entrada:
            w_ini: Punto de inicio
            lr: Tasa de aprendizaje(learning rate)
            hess: Matriz Hessiana con las segundas derivadas
            grad_fun: Gradiente de la función a minimizar
            fun: Función a minimizar(error)
            epsilon: Error Admisible
            max_iters: Número máximo de iteraciones
            
        Salida:
            w: Punto donde se alcanza el supuesto mínimo 
            iterations: Número de iteraciones realizado'''
    iterations = 0
    w = w_ini.copy()
    error_actual = fun(w[0], w[1])
    while max_iters > iterations and error_actual > epsilon:
        w = w - lr * np.linalg.inv(hess(w[0], w[1])).dot(grad_fun(w[0], w[1]))          #El algoritmo solo cambia en esta línea, donde se introduce el producto por la inversa del hessiano
        error_actual = fun(w[0], w[1])
        iterations = iterations + 1
    return w, iterations

def relaxedNewtonRaphson_extrareturns(w_ini, lr, hess, grad_fun, fun, epsilon, max_iters):
    '''Entrada:
            w_ini: Punto de inicio
            lr: Tasa de aprendizaje(learning rate)
            hess: Matriz Hessiana con las segundas derivadas
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
        #print(np.linalg.inv(hess(w[0], w[1])))
        w = w - lr * np.linalg.inv(hess(w[0], w[1])).dot(grad_fun(w[0], w[1]))      
        #print(hess(w[0], w[1]))
        error_actual = fun(w[0], w[1])
        iterations = iterations + 1
        traza.append(w)
        evol.append(error_actual)
    return w, iterations, traza,  evol  


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


#Lanzamos el experimento de igual forma que en el ejercicio 1

eta = 0.01
maxIter = 50
error2get = -np.Infinity
initial_point = np.array([-1.0, 1.0])
w, it, traza, evol = relaxedNewtonRaphson_extrareturns(initial_point, eta, hessf, gradf, f, error2get, maxIter)

print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print(f(w[0], w[1]))

iteraciones = np.arange(it+1)
plt.plot(iteraciones+1, evol, label=r'$\eta = 0.01$')
plt.xlabel('Número de iteraciones')
plt.ylabel('Valor de f')
#plt.title(r'Learning rate $\eta = 0.01$')
plt.title('Punto inicial: (-1.0, 1.0)')

w, it, traza, evol = relaxedNewtonRaphson_extrareturns(initial_point, 0.1, hessf, gradf, f, error2get, maxIter)

iteraciones = np.arange(it+1)
plt.plot(iteraciones+1, evol, label=r'$\eta = 0.1$')
plt.xlabel('Número de iteraciones')
plt.ylabel('Valor de f')
#plt.title(r'Learning rate $\eta = 0.01$')
plt.title('Punto inicial: (-1.0, 1.0)')
plt.legend()

print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print(f(w[0], w[1]))

labels = ['x', 'y', 'f(x, y)']
display_figure(5, f, traza, 'jet', 'Ejercicio 3 : Método de Newton', labels)
plt.show()



#Hacemos lo mismo con el resto de puntos iniciales y tasas, servirá para la comparación

distintosinicios = [np.array([-0.5, -0.5]), np.array([1.0, 1.0]), np.array([2.1, -2.1]), np.array([-3.0, 3.0]), np.array([-2.0, 2.0])]
tasas = [0.01, 0.1]

for ini in distintosinicios:
    for t in tasas:
        w, it, traza, evol = relaxedNewtonRaphson_extrareturns(ini, t, hessf, gradf, f, error2get, maxIter)
        print ('Resultados para punto inicial ', ini, ' y η = ', t, ' : ')
        print ('Numero de iteraciones: ', it)
        print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
        print ('Valor de la función en el mínimo: ' + str(f(w[0], w[1])))


"""

# SE OMITE EL EXPERIMENTO SOBRE LA FUNCIÓN E PORQUE OCURRE LO MISMO Y NO APORTA NADA


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

#Derivada de segundo orden respecto a u

def d2Eu(u, v):
    return v**2 * np.exp(-2*(u**2 + v**2)) * (16*u**4 - 20*u**2 + 2)

#Derivada de segundo orden respecto a u

def d2Ev(u, v):
    return u**2 * np.exp(-2*(u**2 + v**2)) * (16*v**4 - 20*v**2 + 2)

#Derivada de segundo orden cruzada

def d2Euv(u, v):
    return (-4*v**3 + 2*v) * np.exp(-2*(u**2 + v**2)) * (-4*u**3 + 2*u)

#Matriz Hessiana con las derivadas parciales de segundo orden
def hessE(x, y):
    return np.array([d2Eu(x, y), d2Euv(x, y), d2Euv(x, y), d2Ev(x, y)]).reshape((2, 2))  #Hacemos reshape para ponerle la forma matricial habitual del hessiano.


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


eta = 0.01 
maxIter = 100000
error2get = -np.Infinity
initial_point = np.array([0.5, -0.5])
w, it, traza, evol = relaxedNewtonRaphson_extrareturns(initial_point, eta, hessE, gradE, E, error2get, maxIter)

iteraciones = np.arange(it+1)
plt.plot(iteraciones+1, evol)
plt.xlabel('Número de iteraciones')
plt.ylabel('Valor de f')
plt.title(r'Learning rate $\eta = 0.01$')

print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print(f(w[0], w[1]))

labels = ['x', 'y', 'f(x, y)']
display_figure(5, E, traza, 'plasma', 'niuton', labels)
plt.show()
"""