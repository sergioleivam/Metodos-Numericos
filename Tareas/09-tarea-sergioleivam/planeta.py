#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

class Planeta(object):
    """
    Esta clase inicializa un objecto con las instancias de la 
    condicion inicial y el alpha que esta en la correccion del
    potencial gravitacional.
    
     Ademas permite aplicar la ecuacion de movimiento y los 
     metodos de Verlet y RK4, para analizar como cambia en el
     tiempo.
    """

    def __init__(self, condicion_inicial, alpha=0):
        """
        __init__ es un método especial que se usa para inicializar las
        instancias de una clase.

        Ej. de uso:
        >> mercurio = Planeta([x0, y0, vx0, vy0])
        >> print(mercurio.alpha)
        >> 0.
        """
        self.estado_actual = condicion_inicial
        self.t_actual = 0.
        self.alpha = alpha

    def ecuacion_de_movimiento(self):
        """
        Implementa la ecuación de movimiento, como sistema de ecuaciónes de
        primer orden.
        """
        x, y, vx, vy = self.estado_actual
        a = self.alpha
        r = (x**2+y**2)**0.5
        fx = -x * r**(-3) + 2 * a * x * r**(-4)
        fy = -y * r**(-3) + 2 * a * y * r**(-4)
        ec_mov = np.array([vx, vy, fx, fy])
        return ec_mov
    
    def avanza_rk4(self, dt):
        """
        Toma la condición actual del planeta y avanza su posicion y velocidad
        en un intervalo de tiempo dt usando el método de RK4. El método no
        retorna nada, pero modifica los valores de self.estado_actual.
        """
        estado = self.estado_actual
        
        f_k1 = self.ecuacion_de_movimiento()
        k1 = dt*f_k1
        self.estado_actual = estado + k1/2.
        f_k2=self.ecuacion_de_movimiento()
        k2=dt*f_k2        
        self.estado_actual = estado + k2/2.
        f_k3=self.ecuacion_de_movimiento()
        k3=dt*f_k2        
        self.estado_actual = estado + k3
        f_k4=self.ecuacion_de_movimiento()
        k4=dt*f_k4        
        self.estado_actual = estado + (k4 + 2*(k3 + k2) + k1) / 6.
    
    def avanza_verlet(self, dt):
        """
        Similar a avanza_rk4, pero usando Verlet.
        """
        x, y, vx, vy = self.estado_actual
        fx = self.ecuacion_de_movimiento()[2]
        fy = self.ecuacion_de_movimiento()[3]        
        pos = np.array([x, y])
        vel = np.array([vx, vy])
        acel = np.array([fx,fy])
        pos_next = pos + vel * dt + .5 * acel * dt**2    
        self.estado_actual[0:2] = pos_next
        acel_next = self.ecuacion_de_movimiento()[2:4]
        vel_next = vel + .5 * (acel + acel_next) * dt
        y_next = sp.concatenate((pos_next, vel_next))
        self.estado_actual = y_next

    def energia_total(self):
        """
        Calcula la enérgía total del sistema en las condiciones actuales.
        """
        x, y, vx, vy = self.estado_actual
        r = (x**2 + y**2)**(1/2.)
        a=self.alpha
        E= (vx**2 + vy**2)/2. -1 / r + a/ r**2.
        return E
