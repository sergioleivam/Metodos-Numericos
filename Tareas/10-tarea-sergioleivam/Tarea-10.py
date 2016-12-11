from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


# Crea cuadriculado
def cuad(dim_x, dim_y, h):
    '''
    Crea un cuadriculado de dimensiones rectangulares, con dim_x x dim_y
    cada cuadrado con dimensiones de 0.2 cm.
    '''
    x = [i*h for i in range(dim_x)]
    y = [i*h for i in range(dim_y)]

    X, Y = np.meshgrid(x, y)
    return X, Y
# cuad_x, cuad_y = cuad(10, 10, 0.2)
print np.shape(cuad_x)

print "wena fabio"
