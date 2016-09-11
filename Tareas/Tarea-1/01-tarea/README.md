# Tarea nro. 1
## FI3104B - Metodos Numericos Para la Ciencia y la Ingenieria
#### Prof. Valentino Gonzalez

La temperatura effectiva de una estrella corresponde a la temperatura del
cuerpo negro que mejor reproduce su espectro. Por ejemplo, el cuerpo negro que
mejor se ajusta a nuestro Sol, tiene una temperatura aproximada de 5.778 K y
por lo tanto se dice que esa es la temperatura de nuestro Sol.

1. El archivo `sun_AM0.dat` contiene el espectro del Sol medido justo afuera de
   nuestra atmosfera.  El archivo contiene el espectro en unidades de *energia
   por unidad de tiempo por unidad de area por unidad de longitud de onda*. Lea
   el archivo y plotee el espectro de la estrella (es decir, flujo por unidad
   de longitud de onda vs. longitud de onda). Use la convencion astronomica
   para su plot, esto es, usar *cgs* para las unidades de flujo y *Angstrom* o
   *micron* para la longitud de onda.  Recuerde anotar los ejes incluyendo las
   unidades.

    > __Ayuda.__
    >- El modulo `numpy` contiene la rutina `numpy.loadtxt` que le puede ser
    >  util para leer el archivo.
    >- Para plotear se recomienda usar el modulo `matplotlib`. Hay muchos
    >  ejemplos, con codigo incluido en el siguiente
    >  [link](http://matplotlib.org/gallery.html), en particular [este ejemplo
    >  sencillo](http://matplotlib.org/examples/pylab_examples/simple_plot.html)
    >  puede ser util.

2. Elija un metodo apropiado para integrar el espectro en longitud de onda y
   calcule la luminosidad total de la estrella (energia por unidad de tiempo).
   Al integrar obtendra energia por unidad de tiempo por unidad de area. Debe
   multiplicar por 4&pi;d<sup>2</sup> para obtener la energia total por unidad
   de tiempo, donde d es la distancia al Sol (_googlee_).  Se pide que escriba
   su propio algoritmo para llevar a cabo la integracion, mas adelante usaremos
   librerias de libre disposicion.

3. La radiacion de un cuerpo negro en unidades de energia por unidad de tiempo
   por unidad de area por unidad de longitud de onda esta dada por la funcion
   de Planck:

	<img src='eqs/planck.png' alt='Plank' height='70'>

	(latex: $$B_\lambda(T) = \frac{2 \pi h c^2 / \lambda^5}{e^{hc / \lambda k_B T} - 1})

    donde h es la constante de Planck, c es la velocidad de la luz en el vacio,
    k<sub>B</sub> es la constante de Bolzmann, T es la temperatura del cuerpo
    negro y &lambda; es la longitud de onda (esta ecuacion tiene un factor &pi;
    _de mas_ pues estamos interesados en integrar en angulo solido. No se
    preocupe por esto si no quiere, solo use la ecuacion dada).

    Integre numericamente la funcion de Planck para estimar la energia total
    por unidad de tiempo y de area superficial emitida por un cuerpo negro con
    la temperatura efectiva del Sol (escriba su propio algoritmo). Para obtener
    la energia total por unidad de tiempo, debe multiplicar su resultado por
    4&pi;R<sub>eff</sub>; donde R<sub>eff</sub> es el radio effectivo del Sol.
    Compare lo que acaba de calcular con la energia total calculada en 2 para
    estimar el radio efectivo del sol.

    >__Nota__. Se puede demostrar que la integral de la funcion de Planck
    >corresponde a:

    ><img src='eqs/planck_integrated.png' alt='Plank Integrated' height='70'>

	>(latex: $$P = \frac{2 \pi h}{c^2}\left( \frac{k_B T}{h} \right)^4 \int_0^\infty \frac{x^3}{e^x - 1}$$)

    >Y la integral se puede calcular analiticamente con resultado
    >&pi;<sup>4</sup>/15. El problema pide elegir un metodo apropiado y
    >__calcular la integral numericamente__ para luego comparar con el
    >resultado analitico.

    >__Ayuda.__
    >- El modulo `astropy` contiene el submodulo `astropy.constants` que
    >  incluye todas las constantes necesarias ademas de rutinas para cambiar
    >  unidades.
    >- La integral que es necesario calcular es entre 0 e &infin; asi que
    >  requiere ser normalizada. Intente el cambio de variable y=arctan(x) u
    >  otro que le parezca conveniente.
    >- Implemente un algoritmo que permita ir refinando el valor de la integral
    >  con una tolerancia elegida por Ud.

4. El modulo `scipy` en Python incluye los metodos `scipy.integrate.trapz` y
   `scipy.integrate.quad`. Utilicelos para re-calcular las integrales
   calculadas en 2. y 3. Compare los valores obtenidos y la velocidad de
   ejecucion del algoritmo escrito por Ud. vs. `scipy` ¿A que se debe la
   diferencia?

	>__Ayuda.__

    >En la consola `ipython` existe la `ipython magic` `%timeit` que permite
    >estimar velocidad de ejecucion de funciones.


__Otras Notas.__
- Utilice `git` durante el desarrollo de la tarea para mantener un historial de
  los cambios realizados. La siguiente [*cheat
  sheet*](https://education.github.com/git-cheat-sheet-education.pdf) le puede
  ser util. Esto no sera evaluado esta vez pero evaluaremos el uso efectivo de
  git en el futuro, asi que empiece a usarlo.
- La tarea se entrega como un *push* simple a su repositorio privado. El *push*
  debe incluir todos los codigos usados ademas de su informe.
- El informe debe ser entregado en formato *pdf*, este debe ser claro sin
  informacion ni de mas ni de menos. Esto es importante, no escriba de mas,
  esto no mejorara su nota sino que al contrario. Asegúrese de utilizar figuras
  efectivas y tablas para resumir sus resultados. Revise su ortografia.


_Los acentos han sido omitidos a proposito de este documento._
