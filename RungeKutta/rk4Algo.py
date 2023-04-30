def dsbRK4(y0, f, D, n, h):
    """
    This function runs the Runge-Kutta 4th Order method for a D-dimensional system of ODEs

    Parameters
    ----------
    y0: array (of dimension D)
        Initial conditions of the ODEs

    f : function
        The function defining the ODEs of the particular problem of interest.

    D : int
        Dimension (i.e. the amount of coordinates) needed for the system. This can also be thought of as how many different first order ODEs need to be solved.
    n : int
        Number of steps through time

    Returns
    -------
    The velocities of the particular coordinates of interest.
    """
    import numpy as np

    #Initializes the multi-dimensional array for solving the system of ODEs. The first 
    y = np.empty(shape = (n, D))
    t, dt = np.empty([n]), h

    #Setting the initial conditions of the system
    if len(y0) < D: 
        print("The dimension of the initial conditions is less than required")
        return
    elif len(y0)  > D:
        print("The dimension of the initial conditions is more than required")
        return
    else:    
        y[0] = y0

    for i in range(0,n-1):

        #Solving for the coefficients
        k1 = dt*f(t[i], y[i])
        k2 = dt*f(t[i] + 0.5*dt, y[i] + 0.5*k1)
        k3 = dt*f(t[i] + 0.5*dt, y[i] + 0.5*k2)
        k4 = dt*f(t[i] + dt, y[i] + k3)

        #Iteraing through the function and time values
        y[i+1] = y[i] + (1/6)*k1 + (1/3)*k2 + (1/3)*k3 + (1/6)*k4
        t[i+1] = t[i] + dt

    return y.transpose(), t