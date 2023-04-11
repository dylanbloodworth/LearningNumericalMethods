import numpy as np
import matplotlib.pyplot as plt

#These are the First order ODEs that would need to be solved by the Runge-Kutta method

def f(t, y):

    l = y.size
    v = np.empty([l])

    ## This is where the system of ODEs would go. You would have them written as v[0], v[1], ...

    q, b, omega0 = 0.5, 0.85, 2/3

    v[0] = y[1]
    v[1] = -q*y[1] -np.sin(y[0]) + b * np.cos(omega0 * t)

    return v

D = 2
n = 10000

y = np.empty(shape = (n+1, D))
t, dt = np.empty([n+1]), (3*np.pi)/10

y[0] = [0, 2]

for i in range(0,n):

    k1 = dt*f(t[i], y[i])
    k2 = dt*f(t[i] + 0.5*dt, y[i] + 0.5*k1)
    k3 = dt*f(t[i] + 0.5*dt, y[i] + 0.5*k2)
    k4 = dt*f(t[i] + dt, y[i] + k3)
    y[i+1] = y[i] + (1/6)*k1 + (1/3)*k2 + (1/3)*k3 + (1/6)*k4
    t[i+1] = t[i] + dt

y1 = y[:,0]
y2 = y[:,1]


plt.style.use('seaborn-white')
plt.scatter(y1, y2) 
pi = np.pi
plt.xticks(np.arange(-pi, pi+pi/2, step=(pi/2)), ['-π','-π/2','0','π/2','π'])
plt.title("Periodic Behavior of a Driven Pendulum")
plt.xlabel("$\\theta$")
plt.ylabel("$\omega$")
plt.show()