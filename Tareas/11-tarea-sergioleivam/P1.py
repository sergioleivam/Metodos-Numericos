from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import scipy.linalg as alg
import matplotlib.cm as cm


# Constantes
N_tot = 500
gamma = 0.001
mu = 1.5
# Tiempo final
t_fin = 4
# Pasos
dt = 1e-2
dx = 1 / N_tot
# Espaciados
x = np.linspace(0, 1, N_tot)
t = np.linspace(0, 4, 4 / dt)
# Funcion vacia
T = 5 / dt
X = 1 / dx
n = [np.zeros(N_tot) for i in range(int(T))]
# Constantes para integrar
r = gamma * dt / (2 * dx**2)
a = r / (1 + 2 * r)
b = (1 + 2 * r) / (1 - 2 * r)
c = dt * mu

# Condiciones de borde


def inicia_cond_bord(X, T):
    '''
    Toma los valores X , T y dx, numero total de valores en x y t
    respectivamente, por lo tanto ajusta el set de condiciones iniciales a una
    matriz o arreglo de dimensiones genericas (X,T).
     El parametro dx permite dar un espaciado de x generico.
    La funcion solo toma la variable global n, y le asigna los valores de
    las condiciones de borde.
    '''
    for i in range(int(T)):
        n[i][0] = 1
        # No es necesario por la forma en que esta hecho n
        n[i][int(X) - 1] = 0
    for i in range(int(X)):
        n[0][i] = np.exp(- (i * dx)**2 / 0.1)


def resuelve_fila(i, matriz):
    '''
    Esta funcion toma un valor i, entero, que indica que fila temporal se
    quiere calcular,a partir de la fila temporal actual (fila i-1).
     Crea una matriz llena de ceros, y los rellena con los coeficientes que
     se definen a partir del metodo de Crank-Nicolson, para la parte de
     difusion y el metodo de Euler explicito para la parte de reaccion.
    Esta funcion no retorna nada, solo modifica la fila i-esima de la matriz n.
    '''
    n_actual = n[i - 1]
    vector_sol = np.zeros(N_tot)
    vector_sol[0] = n_actual[0]
    vector_sol[N_tot - 1] = n_actual[N_tot - 1]
    for j in range(1, N_tot - 1):
        vector_sol[j] = r * n_actual[j + 1] + (1 - 2 * r) * n_actual[j]
        + r * n_actual[
            j - 1] + mu * dt * (n_actual[j] - (n_actual[j])**2)
    n[i] = alg.solve(matriz, vector_sol)
    n[i][0] = 1
    n[i][N_tot - 1] = 0


# Crea la matriz con los coeficientes
def matriz(N_tot):
    matriz = [np.zeros(N_tot) for i in range(N_tot)]
    matriz[0][0] = 1
    for j in range(1, N_tot - 1):
        matriz[j][j - 1] = -r
        matriz[j][j] = 1 + 2 * r
        matriz[j][j + 1] = -r
    matriz[N_tot - 1][N_tot - 1] = 1
    return matriz

matriz = matriz(N_tot)
fila = int(T) - 1
colum = 0
inicia_cond_bord(X, T)
# print n[fila][colum]

# Resuelve para la T filas temporales
for i in range(1, int(T)):
    resuelve_fila(i, matriz)

# Grafica en 3-D
"""
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
X1, T1 = np.meshgrid(x, t)
ax2.pcolormesh(X1, T1, n, cmap=cm.Greys)


plt.draw()
plt.xlabel('espaciado de x')
plt.ylabel('espaciado de t')
plt.title('Solucion ec. Fisher-KPP, dt=0.01')
"""
# Grafica x,n en varios t
for i in range(0, int(T), 50):
    plt.plot(x, n[i], label="t=" + str(i * dt))
print int(T)
# plt.ylim(0, 1)
plt.xlabel("Posicion en el espacio $x$ ")
plt.ylabel("Densidad de la especie $n$ ")
plt.title("Grafico de densidad v posicion, entre t=0 y t=4.5")
plt.legend(loc='lower left')
plt.savefig("p_1.png")
plt.show()
