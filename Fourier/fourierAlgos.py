from numpy import array, ndarray, size, zeros, pi, exp, sqrt, complex128


def DFT(y: 'ndarray', N : 'int', h : 'float'):
    """
    Parameters
    ----------
    y : ndarray
        This is the array of the function values y(t) that are used to calculate the Fourier Transform Y(omega)
    N : int
        Length of the y array
    h : float
        Step size use to calculate the y values
        
    Returns
    -------
    N_list : ndarray
        The omega values
    Y : ndarray
        The absolute value of the imaginary component of the Fourier Transform
    """
    N = size(y)

    Y = zeros((N,), dtype= complex128)
    N_list = zeros((N,))

    for n in range(0,N):    
        for k in range(0,N): 
            Y[n] += y[k]*exp(-pi*2j*n*k/N)
        
        Y[n] = (1/sqrt(2*pi))*Y[n]
        N_list[n] = n*((2*pi)/(N*h))

    return array(N_list), array(Y)


def invDFT(Y:'list', h:'float'):
    """
    Parameters
    ----------
    Y : ndarray
        This is the array of the function values y(t) that are used to calculate the Fourier Transform Y(omega)

    h : float
        uniform time step used in the Fourier Transform
        
    Returns
    -------
    N_list : ndarray
        The omega values
    Y : ndarray
        The absolute value of the imaginary component of the Fourier Transform
    """
    N = size(Y)
    y = zeros((N,), complex128)
    t = zeros((N,))

    for k in range(0,N):
        for n in range(0, N):
            y[k] += Y[n]*exp(pi*2j*n*k/N)
        
        y[k] = y[k]*((sqrt(2*pi)/N))
        t[k] = k*h

    return t, y.real
