from math import cos, pi
from sympy import Poly, symbols, lambdify
import numpy as np

def dividedDiffernce(f, xlow, xhigh, deg=4, var_name='x'):
    """
    INPUT:
        f: The function to be interpolated as SymPy expression.
        xlow: The lower bound of the interval.
        xhigh: The upper bound of the interval.
        var_name(default='x'): The name of the varible of the
            interpolating polynomial.
        
    OUTPUT:
        P: An n degree polynomial that interpolates the function
            f with respect to the points (x_i, y_i), where n is
            the number of points minus one.
        R_bar_squared: The adjusted R-squared coefficient for the
            polynomial fit.
    
    Performs Newton's Divided Differences method to interpolate a function f, 
    with respect to the x-values, x_0, x_1, ..., x_n, on the
    interval [xlow, xhigh]. These values are taken to be the n+1 Chebyshev points 
    mapped to the given interval. The interpolating polynomaial P_n(x) takes the form,

    P_n(x) = a_0 + a_1*(x-x_0) + a_2*(x-x_0)*(x-x_1)
            + ... + a_n*(x-x_0)*...*(x-x_n-1),

    where each a_k is the kth divided difference coefficient. We can solve
    for each a_k by using the following notation,

    a_k = f[x_0,x_1,...,x_k],

    where each f[x_0,x_1,...,x_k] is recursivly calculated using the
    two previous coefficients as,

    f[x_i,x_i+1,...,x_i+k] = (f[x_i+1,x_i+2,...,x_i+k]
                            - f[x_i,x_i+1,...,x_i+k-1]) / (x_i+k - x_i),

    starting with f[x_0] = f(x_0).

    The recursion can be illustrated in the following way,

    f[x_0]   
    f[x_1]   f[x_0,x_1]
    f[x_2]   f[x_1,x_2]   f[x_0,x_1,x_2]
    f[x_3]   f[x_2,x_3]   f[x_1,x_2,x_3] 
    f[x_4]   f[x_3,x_4]   f[x_2,x_3,x_4] 
    .         .            .               .
    .         .            .                 .
    .         .            .                   .
    f[x_n]   f[x_n-1,x_n] f[x_n-2,x_n-1,x_n] ... f[x_0,x_,...,x_n]

    where the kth coefficient along the diagonal is equal to a_k. Using
    the n coefficients, the n degree polynomial P_n(x) is generated
    for the set of given x values.
    """

    NUM_NODES = deg + 1
    xvals = chebNodes(xlow, xhigh, NUM_NODES)
    yvals = [f.subs(var_name, xi) for xi in xvals]
    
    x = symbols(var_name)
    
    # The 2-D list of divided-difference coefficients 
    # (The coefficients that correspond to each a_k will be along 
    # the diagonal of the list.
    coeffs = []
    
    # Fill each element in the first column, with the corresponding y 
    # value, i.e., coeffs_{i,0} = f(x_{i}).
    for row in range(NUM_NODES):
        coeffs.append([])
        coeffs[row].append(yvals[row])

    # Iterate through every other column, and recursivly generate the 
    # divided-difference coefficients.
    for col in range(1, NUM_NODES):
        for row in range(col, NUM_NODES):
            # coeffs_{i,j} = (coeffs_{i,j-1} - coeffs_{i-1,j-1}) / (x_{i} - x_{i-j})
            coeffs[row].append((coeffs[row][col-1] - coeffs[row-1][col-1])/(xvals[row]
                                                             - xvals[row-col]))

    # Using the coefficients, a_{n} = coeffs_{i} = coeffs[i][i], construct the 
    # interpolating polynomial P(x), such that...,
    # P_{n}(x) = coeffs_{0} + coeffs_{1}*(x - x_{0}) + coeffs_{2}*(x - x_{0})*(x - x_{1})
    #           + ... + coeffs_{n}*(x - x_{0})*(x - x_{1})*...(x - x_{n-1})
    P = Poly([coeffs[0][0]], x)
    for i in range(1, NUM_NODES):
        term = coeffs[i][i]
        for k in range(i):
            term *= (x - xvals[k])

        P += term

    # Calculate the adjusted R-squared, using 50 test points.
    R_bar_sqr = _getRbarSquared(f, P, var_name, xlow, xhigh, n_steps=50)
    
    return P, R_bar_sqr


def chebNodes(xlow, xhigh, npoints):
    """
    Returns a set of Chebyshev nodes on the interval [xlow, xhigh]. 
    """

    cheb_nodes = [_kNode(i, npoints) for i in range(npoints)]
    
    # Translate nodes to the interval xlow to xhigh. 
    cheb_nodes = list(map(lambda c: 0.5*(xlow + xhigh)
                          + 0.5*(xhigh - xlow)*c, cheb_nodes))
    cheb_nodes.sort()
    return cheb_nodes


def _kNode(k, n):
    """ 
    Returns the kth Chebyshev node for a given number of n
    points, for an interval of -1 to 1.
    """
    return cos((2*k + 1)*pi/(2*(n - 1) + 2))


def _getRbarSquared(f, polyfit, var_name, xmin, xmax, n_steps=50):
    endpoint_shift = 0.0001           # To ensure all domian boundaries yield real y values.
    step = abs(xmin - xmax)/n_steps                     
    x = np.arange(xmin + endpoint_shift, xmax - endpoint_shift, step)
    
    lam_f = lambdify(var_name, f, 'numpy')
    lam_poly = lambdify(var_name, polyfit.as_expr(), 'numpy')
    y = lam_f(x)
    yfit = lam_poly(x)
    
    yfit = yfit[np.isreal(y)]
    y = y[np.isreal(y)]

    assert all([not np.isnan(yi) for yi in y]), 'Floating point arithmatic round off \
    error in evaluation of y-values causing complex y.'
    
    y_resid = list(map(lambda yi, yfi: yi - yfi, y, yfit))

    SSresid = sum(map(lambda r: r**2, y_resid))
    SStotal = (len(y) - 1)*np.var(y)

    rsq_bar = 1 - SSresid/SStotal*(len(y) - 1)/(len(y) - polyfit.degree() - 1)
    return float(rsq_bar)
















