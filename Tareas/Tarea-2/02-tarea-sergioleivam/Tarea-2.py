
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.optimize
get_ipython().magic(u'matplotlib inline')


### codigo de contour, subido por el profe 
### solo grafica el nivel cero, para mi facilidad

#### 

x = np.linspace(-3, 3, 300)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)
F1 = X**4 + Y**4-15
F2=X**3*Y - X*Y**3 - Y/2 - 1.937
plt.clf()

levels = [0]
plt.contour(X, Y, F1, levels)
plt.contour(X, Y, F2, levels)

plt.xlabel('$x$', fontsize=18)
plt.ylabel('$y$', fontsize=18)
plt.title('Funciones F1 y F2 en nivel cero')


######



#### Implementacion de la parametrizacion
#### Con parte del codigo de contour, mas la parametrizacion
t=np.linspace(-np.pi,np.pi,300)
xt=15**(1/4.)*np.sign(np.sin(t))*(np.sin(t)*np.sin(t))**(1/4.)
yt=15**(1/4.)*np.sign(np.cos(t))*(np.cos(t)*np.cos(t))**(1/4.)


X, Y = np.meshgrid(xt, yt)
F1 = X**4 + Y**4-15
F2=X**3*Y - X*Y**3 - Y/2 - 1.937
plt.clf()

levels = [0]

plt.contour(X, Y, F1, levels)
plt.contour(X, Y, F2, levels)

plt.xlabel('$x$', fontsize=18)
plt.ylabel('$y$', fontsize=18)

plt.xlim(-2.5,2.5)
plt.ylim(-2.5,2.5)

plt.title('Funciones F1 y F2 en nivel cero, con parametrizacion')

## definimos la funcion
## para llevar la funcion F2 de dos variables (x,t), a una de 1 variable (t)

def f2(to):
    xt=15**(1/4.)*np.sign(np.sin(to))*(np.sin(to)*np.sin(to))**(1/4.)
    yt=15**(1/4.)*np.sign(np.cos(to))*(np.cos(to)*np.cos(to))**(1/4.)
    F2=xt**3*yt - xt*yt**3 - yt/2 - 1.937
    return F2



#Dado que no era obligatorio hacer un algortimo para en encontrar los intervalos de [a,b], no lo hice y los busque a mano con la siguiente "rutina"

a=scipy.optimize.bisect(f2,-2.2 , -2.3  )
t=np.linspace(-np.pi,np.pi,300.)
punto=-2.2
print 'funcion',f2(punto),'punto', punto,
print 'scipy' , a 
print -2.2 , -2.3


plt.plot(t,f2(t))

plt.xlabel('t', fontsize=18)
plt.ylabel('f2(t)', fontsize=18)
plt.title('f2 vs parametrizacion')

plt.grid('on') 



