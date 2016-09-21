
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.optimize
get_ipython().magic(u'matplotlib inline')


# In[2]:

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


# In[3]:

#### Implementacion de la parametrizacion 
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


# In[4]:

## definimos la funcion 

def f2(to):
    #t=np.linspace(-np.pi,np.pi,300.)
    xt=15**(1/4.)*np.sign(np.sin(to))*(np.sin(to)*np.sin(to))**(1/4.)
    yt=15**(1/4.)*np.sign(np.cos(to))*(np.cos(to)*np.cos(to))**(1/4.)
    j=0
    #e=2*np.pi/300
    #for i in range(np.shape(t)[0]):
    #    if abs(to-i)<=e:
    #        j=i
    F2=xt**3*yt - xt*yt**3 - yt/2 - 1.937
    return F2



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



