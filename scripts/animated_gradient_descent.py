import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
import matplotlib.animation as animation

x = np.arange(-10, 10, 0.2)
y = x**2 + 2*x

fig= plt.figure()
axes=fig.add_subplot(111)
axes.set_xlim([-11, 11])
axes.set_ylim([-10, 150])
axes.plot(x,y)

axes.annotate(xy=(-8, 130), s = '$f(x) = x^2 + 2x$', fontsize=18)
axes.annotate(xy=(-8, 115), s = '$df(x)/dx = 2x + 2$', fontsize=18)

plt.show()


# Steigung an Punkt x
f = lambda x : x**2 + 2*x
f_ = lambda x : 2*x + 2

w = [8.5]

def init():
    return []

alpha = 0.2
iter = 1

def run(j):    
    if j > 1:
        axes.plot([w[-2]-3, w[-2]+3], [f(w[-2]) - 3*f_(w[-2]), f(w[-2]) + 3*f_(w[-2])], color='white',
              linestyle = '-')
        axes.plot([w[-2]-3, w[-2]+3], [f(w[-2]) - 3*f_(w[-2]), f(w[-2]) + 3*f_(w[-2])], color='grey',
              linestyle = '--', alpha = 0.5 )
    axes.plot([w[-1]-3, w[-1]+3], [f(w[-1]) - 3*f_(w[-1]), f(w[-1]) + 3*f_(w[-1])], color='orange',
              linestyle = '-')

        
    axes.plot(w[-1], f(w[-1]), 'ro')
    new_w = w[-1] - alpha * f_(w[-1])

    if abs(new_w -w[-1]) < 0.2:
        return
    w.append(new_w)
    plt.quiver(w[-2], f(w[-2])+ 5, -alpha * f_(w[-2]), f(w[-1])- f(w[-2]),
               scale_units='xy', angles='xy', scale=1)

ani = animation.FuncAnimation(fig, func = run, init_func = init,
                              frames = (i for i in np.range(10)),
                       interval = 1000,  blit = False) 
