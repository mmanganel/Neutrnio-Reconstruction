from sympy import symbols, solve, sqrt
import numpy as np
from neurecon.interpolation import dividedDiffernce

def reconstruct(edata, mwm=80.4, cme=1000):
    """
        Reconstructs the momentum of the neutrino and anti-neutrino, given the
        momentum of the muons and bottom quarks. 
        INPUT:
            edata: A list containing the x, y, and z momentum in GeV of the charged leptons
            and bottom quarks, in the following order: 
                edata := [amux, amuy, amuz, b1x, b1y, b1z, mux, muy, muz, b2x, b2y, b2z]
            with notation, 
                amu := anti-muon
                b1 := bottom quark 1*
                mu := muon
                b2 := bottom quark 2*
                * The charge of the bottom quark is assumed to be unknown.

            mwm(default=80.4): The constrained mass of the W boson in GeV.

            cme(default=1000): The center of mass energy.
        
        OUTPUT:
            solutions: A list of the reconstructed neutrino and anti-neutrino 
            x, y, and z-momenta as a tuple, for each possible solution of p2z, 
            [(nux, nuy, nuz, anux, anuy, anuz), ...].

    """
    
    assert len(edata) == 12, 'edata should be a list of length 12, containing \
    the mommenta of the muon and bottom quarks.'
    
    degree = 4                  # The degree of the interpolating polynomial.
    rbar_threshold = 0.95     
    mwm2 = mwm**2
    
    domain_func, func1s, func2s = _getFuncs(edata, mwm2, cme)
    p2z_func1 = func1s[2]
    p2z_func2 = func2s[2]
    solutions = []

    # Find domain by finding the two roots of domain_func (quadratic).
    domain = solve(domain_func, rational=False,
                   simplify=False, minimal=True, quadratics=True)

    # Check for complex domain bounds.
    if not any([d.is_real for d in domain]): 
        return [], (None, None)
    
    domain = [float(d) for d in domain]

    # Interpolate function 1 and calculate adjusted R-squared (rbar). 
    poly1, rbar1 = dividedDiffernce(p2z_func1, domain[0], domain[1],
                              deg=degree, var_name='p2z')
    # Add solutions only if interpolation is a good fit.
    if rbar1 > rbar_threshold:
        solutions = _getSols(poly1, domain, func1s)

    # Interpolate function 2 and calculate adjusted R-squared.
    poly2, rbar2 = dividedDiffernce(p2z_func2, domain[0], domain[1],
                              deg=degree, var_name='p2z')
    
    if rbar2 > rbar_threshold: 
        solutions += _getSols(poly2, domain, func2s)
    
#    rbars = (rbar1, rbar2)
    return solutions


def _getSols(poly, domain, funcs):
    # Computes the real roots to the interpolating polynomial 
    # within the given domain, and returns a list of the x, y, 
    # and z components of the neutrinos calculated for each solution. 
    
    cmp_sols = []
    
    # Find the roots of the interpolating polynomial.
    roots = np.roots(poly.all_coeffs())     

    # Take only real roots, within the function domain.
    roots = [float(r.real) for r in roots if abs(r.imag/r.real) < 1e-7]     
    sols = [r for r in roots if r >= domain[0] and r <= domain[1]]

    # Evaluate the other neutrino and anti-neutrino components for each solution.
    for sol in sols:
        cmp_sols.append(_evalComponents(sol, funcs))
        
    return cmp_sols


def _evalComponents(p2z_sol, cmp_funcs):
    # Calculates the unknown components of the neutrino and anti-neutrino 
    # for a given solution. 
    
        # Evaluate neutrino component functions. 
        fsols = [f.subs('p2z', p2z_sol) for f in cmp_funcs[:2] + cmp_funcs[3:]]
        fsols.insert(2, p2z_sol)
        
        fsols = [float(s) for s in fsols]
        
        return tuple(fsols)


def _getFuncs(edata, mwm2, cme):
    
# p1 := anti-muon
# p2 := neutrino
# p3 := bottom quark 1*
# p4 := muon
# p5 := anti-neutrino
# p6 := bottom quark 2*
# * The charge of the bottom quark is assumed to be unknown.           
    
    p1x, p1y, p1z = edata[:3]
    p3x, p3y, p3z = edata[3:6]
    p4x, p4y, p4z = edata[6:9]
    p6x, p6y, p6z = edata[9:12]

    # Remaining momentum.
    premx = - p1x - p3x - p4x - p6x     
    premy = - p1y - p3y - p4y - p6y
    premz = - p1z - p3z - p4z - p6z

    # Remaining energy.
    Erem = (cme - np.sqrt(p1x**2 + p1y**2 + p1z**2)   
            - np.sqrt(p3x**2 + p3y**2 + p3z**2)
            - np.sqrt(p4x**2 + p4y**2 + p4z**2)
            - np.sqrt(p6x**2 + p6y**2 + p6z**2))

    p4x2, p4x3, p4x4 = [p4x**i for i in range(2,5)]     # Powers of p4x.
    p4y2, p4y3, p4y4 = [p4y**i for i in range(2,5)]     # Powers of p4y.
    p4z2, p4z3, p4z4 = [p4z**i for i in range(2,5)]     # Powers of p4z.

    premx2, premx3, premx4, premx5, premx6 = [premx**i for i in range(2,7)]     # Powers of premx.
    premy2, premy3, premy4, premy5, premy6 = [premy**i for i in range(2,7)]     # Powers of premy.
    premz2, premz3, premz4 = [premz**i for i in range(2,5)]                     # Powers of premz.
    Erem2, Erem3, Erem4, Erem5, Erem6 = [Erem**i for i in range(2,7)]           # Powers of Erem.

    p4_mag = np.sqrt(p4x2 + p4y2 + p4z2)

    p2z = symbols('p2z')
 
    A = (4*Erem4*p4x4+8*Erem4*p4x2*p4y2+4*Erem4*p4y4-8*Erem2*p4x4*premx2
        -8*Erem2*p4x2*p4y2*premx2-8*Erem2*p4x2*p4z2*premx2+8*Erem2*p4y2*p4z2*premx2+4*p4x4*premx4+8*p4x2*p4z2*premx4+4*p4z4*premx4
        -16*Erem2*p4x3*p4y*premx*premy-16*Erem2*p4x*p4y3*premx*premy-32*Erem2*p4x*p4y*p4z2*premx*premy+16*p4x3*p4y*premx3*premy+16
        *p4x*p4y*p4z2*premx3*premy-8*Erem2*p4x2*p4y2*premy2-8*Erem2*p4y4*premy2+8*Erem2*p4x2*p4z2*premy2-8*Erem2*p4y2*p4z2*premy2
        +24*p4x2*p4y2*premx2*premy2+8*p4x2*p4z2*premx2*premy2+8*p4y2*p4z2*premx2*premy2+8*p4z4*premx2*premy2+16*p4x*p4y3*premx*premy3
        +16*p4x*p4y*p4z2*premx*premy3+4*p4y4*premy4+8*p4y2*p4z2*premy4+4*p4z4*premy4)

    T = (4*Erem4*mwm2*p4x3+4*Erem4*mwm2*p4x*p4y2-8*Erem4*p2z*p4x3*p4z-8*Erem4*p2z*p4x*p4y2*p4z-4*Erem5*p4x3*p4_mag-4*Erem5*p4x
        *p4y2*p4_mag-4*Erem4*p4x4*premx-4*Erem4*p4x2*p4y2*premx-4*Erem4*p4x2*p4z2*premx+4*Erem4*p4y2*p4z2*premx+4*Erem3*mwm2*p4x2
        *p4_mag*premx-4*Erem3*mwm2*p4y2*p4_mag*premx-8*Erem3*p2z*p4x2*p4z*p4_mag*premx+8*Erem3*p2z*p4y2*p4z*p4_mag*premx-4*Erem2
        *mwm2*p4x3*premx2-8*Erem2*mwm2*p4x*p4y2*premx2+8*Erem2*p2z*p4x3*p4z*premx2+16*Erem2*p2z*p4x*p4y2*p4z*premx2-4*Erem2*mwm2
        *p4x*p4z2*premx2+8*Erem2*p2z*p4x*p4z3*premx2+8*Erem3*p4x3*p4_mag*premx2+4*Erem3*p4x*p4y2*p4_mag*premx2+4*Erem3*p4x*p4z2*p4_mag
        *premx2+8*Erem2*p4x4*premx3+4*Erem2*p4x2*p4y2*premx3+12*Erem2*p4x2*p4z2*premx3-4*Erem2*p4y2*p4z2*premx3+4*Erem2*p4z4*premx3
        -4*Erem*mwm2*p4x2*p4_mag*premx3+8*Erem*p2z*p4x2*p4z*p4_mag*premx3-4*Erem*mwm2*p4z2*p4_mag*premx3+8*Erem*p2z*p4z3*p4_mag*premx3
        -4*Erem*p4x3*p4_mag*premx4-4*Erem*p4x*p4z2*p4_mag*premx4-4*p4x4*premx5-8*p4x2*p4z2*premx5-4*p4z4*premx5-4*Erem4*p4x3*p4y
        *premy-4*Erem4*p4x*p4y3*premy-8*Erem4*p4x*p4y*p4z2*premy+8*Erem3*mwm2*p4x*p4y*p4_mag*premy-16*Erem3*p2z*p4x*p4y*p4z*p4_mag
        *premy+4*Erem2*mwm2*p4x2*p4y*premx*premy-4*Erem2*mwm2*p4y3*premx*premy-8*Erem2*p2z*p4x2*p4y*p4z*premx*premy+8*Erem2*p2z*p4y3
        *p4z*premx*premy-8*Erem2*mwm2*p4y*p4z2*premx*premy+16*Erem2*p2z*p4y*p4z3*premx*premy+8*Erem3*p4x2*p4y*p4_mag*premx*premy
        +8*Erem3*p4y*p4z2*p4_mag*premx*premy+16*Erem2*p4x3*p4y*premx2*premy+4*Erem2*p4x*p4y3*premx2*premy+20*Erem2*p4x*p4y*p4z2*premx2
        *premy+4*mwm2*p4x2*p4y*premx3*premy-8*p2z*p4x2*p4y*p4z*premx3*premy+4*mwm2*p4y*p4z2*premx3*premy-8*p2z*p4y*p4z3*premx3*premy
        -8*Erem*p4x2*p4y*p4_mag*premx3*premy-8*Erem*p4y*p4z2*p4_mag*premx3*premy-12*p4x3*p4y*premx4*premy-12*p4x*p4y*p4z2*premx4
        *premy-4*Erem2*mwm2*p4x3*premy2+8*Erem2*p2z*p4x3*p4z*premy2+4*Erem2*mwm2*p4x*p4z2*premy2-8*Erem2*p2z*p4x*p4z3*premy2+4*Erem3
        *p4x3*p4_mag*premy2+8*Erem3*p4x*p4y2*p4_mag*premy2-4*Erem3*p4x*p4z2*p4_mag*premy2+4*Erem2*p4x4*premx*premy2+16*Erem2*p4x2
        *p4y2*premx*premy2+8*Erem2*p4x2*p4z2*premx*premy2+4*Erem2*p4z4*premx*premy2-8*Erem*mwm2*p4x2*p4_mag*premx*premy2+4*Erem*mwm2
        *p4y2*p4_mag*premx*premy2+16*Erem*p2z*p4x2*p4z*p4_mag*premx*premy2-8*Erem*p2z*p4y2*p4z*p4_mag*premx*premy2-4*Erem*mwm2*p4z2
        *p4_mag*premx*premy2+8*Erem*p2z*p4z3*p4_mag*premx*premy2-4*mwm2*p4x3*premx2*premy2+8*mwm2*p4x*p4y2*premx2*premy2+8*p2z*p4x3
        *p4z*premx2*premy2-16*p2z*p4x*p4y2*p4z*premx2*premy2-4*mwm2*p4x*p4z2*premx2*premy2+8*p2z*p4x*p4z3*premx2*premy2-4*Erem*p4x3
        *p4_mag*premx2*premy2-4*Erem*p4x*p4y2*p4_mag*premx2*premy2-4*p4x4*premx3*premy2-12*p4x2*p4y2*premx3*premy2-12*p4x2*p4z2*premx3
        *premy2-4*p4y2*p4z2*premx3*premy2-8*p4z4*premx3*premy2+4*Erem2*p4x3*p4y*premy3+8*Erem2*p4x*p4y3*premy3+12*Erem2*p4x*p4y*p4z2
        *premy3-8*Erem*mwm2*p4x*p4y*p4_mag*premy3+16*Erem*p2z*p4x*p4y*p4z*p4_mag*premy3-8*mwm2*p4x2*p4y*premx*premy3+4*mwm2*p4y3
        *premx*premy3+16*p2z*p4x2*p4y*p4z*premx*premy3-8*p2z*p4y3*p4z*premx*premy3+4*mwm2*p4y*p4z2*premx*premy3-8*p2z*p4y*p4z3*premx
        *premy3-8*Erem*p4x2*p4y*p4_mag*premx*premy3-8*Erem*p4y*p4z2*p4_mag*premx*premy3-12*p4x3*p4y*premx2*premy3-4*p4x*p4y3*premx2
        *premy3-16*p4x*p4y*p4z2*premx2*premy3-4*mwm2*p4x*p4y2*premy4+8*p2z*p4x*p4y2*p4z*premy4-4*mwm2*p4x*p4z2*premy4+8*p2z*p4x*p4z3
        *premy4-4*Erem*p4x*p4y2*p4_mag*premy4+4*Erem*p4x*p4z2*p4_mag*premy4-12*p4x2*p4y2*premx*premy4-4*p4x2*p4z2*premx*premy4-4
        *p4y2*p4z2*premx*premy4-4*p4z4*premx*premy4-4*p4x*p4y3*premy5-4*p4x*p4y*p4z2*premy5+8*Erem4*p4x3*p4z*premz+8*Erem4*p4x*p4y2
        *p4z*premz+8*Erem3*p2z*p4x3*p4_mag*premz+8*Erem3*p2z*p4x*p4y2*p4_mag*premz+8*Erem2*p2z*p4x4*premx*premz+8*Erem2*p2z*p4x2
        *p4y2*premx*premz+8*Erem2*p2z*p4x2*p4z2*premx*premz-8*Erem2*p2z*p4y2*p4z2*premx*premz+8*Erem3*p4x2*p4z*p4_mag*premx*premz
        -8*Erem3*p4y2*p4z*p4_mag*premx*premz-8*Erem2*p4x3*p4z*premx2*premz-16*Erem2*p4x*p4y2*p4z*premx2*premz-8*Erem2*p4x*p4z3*premx2
        *premz-8*Erem*p2z*p4x3*p4_mag*premx2*premz-8*Erem*p2z*p4x*p4z2*p4_mag*premx2*premz-8*p2z*p4x4*premx3*premz-16*p2z*p4x2*p4z2
        *premx3*premz-8*p2z*p4z4*premx3*premz-8*Erem*p4x2*p4z*p4_mag*premx3*premz-8*Erem*p4z3*p4_mag*premx3*premz+8*Erem2*p2z*p4x3
        *p4y*premy*premz+8*Erem2*p2z*p4x*p4y3*premy*premz+16*Erem2*p2z*p4x*p4y*p4z2*premy*premz+16*Erem3*p4x*p4y*p4z*p4_mag*premy
        *premz+8*Erem2*p4x2*p4y*p4z*premx*premy*premz-8*Erem2*p4y3*p4z*premx*premy*premz-16*Erem2*p4y*p4z3*premx*premy*premz-16*Erem
        *p2z*p4x2*p4y*p4_mag*premx*premy*premz-16*Erem*p2z*p4y*p4z2*p4_mag*premx*premy*premz-24*p2z*p4x3*p4y*premx2*premy*premz-24
        *p2z*p4x*p4y*p4z2*premx2*premy*premz+8*p4x2*p4y*p4z*premx3*premy*premz+8*p4y*p4z3*premx3*premy*premz-8*Erem2*p4x3*p4z*premy2
        *premz+8*Erem2*p4x*p4z3*premy2*premz-8*Erem*p2z*p4x*p4y2*p4_mag*premy2*premz+8*Erem*p2z*p4x*p4z2*p4_mag*premy2*premz-24*p2z
        *p4x2*p4y2*premx*premy2*premz-8*p2z*p4x2*p4z2*premx*premy2*premz-8*p2z*p4y2*p4z2*premx*premy2*premz-8*p2z*p4z4*premx*premy2
        *premz-16*Erem*p4x2*p4z*p4_mag*premx*premy2*premz+8*Erem*p4y2*p4z*p4_mag*premx*premy2*premz-8*Erem*p4z3*p4_mag*premx*premy2
        *premz-8*p4x3*p4z*premx2*premy2*premz+16*p4x*p4y2*p4z*premx2*premy2*premz-8*p4x*p4z3*premx2*premy2*premz-8*p2z*p4x*p4y3*premy3
        *premz-8*p2z*p4x*p4y*p4z2*premy3*premz-16*Erem*p4x*p4y*p4z*p4_mag*premy3*premz-16*p4x2*p4y*p4z*premx*premy3*premz+8*p4y3
        *p4z*premx*premy3*premz+8*p4y*p4z3*premx*premy3*premz-8*p4x*p4y2*p4z*premy4*premz-8*p4x*p4z3*premy4*premz-4*Erem3*p4x3*p4_mag
        *premz2-4*Erem3*p4x*p4y2*p4_mag*premz2-4*Erem2*p4x4*premx*premz2-4*Erem2*p4x2*p4y2*premx*premz2-4*Erem2*p4x2*p4z2*premx*premz2
        +4*Erem2*p4y2*p4z2*premx*premz2+4*Erem*p4x3*p4_mag*premx2*premz2+4*Erem*p4x*p4z2*p4_mag*premx2*premz2+4*p4x4*premx3*premz2
        +8*p4x2*p4z2*premx3*premz2+4*p4z4*premx3*premz2-4*Erem2*p4x3*p4y*premy*premz2-4*Erem2*p4x*p4y3*premy*premz2-8*Erem2*p4x*p4y
        *p4z2*premy*premz2+8*Erem*p4x2*p4y*p4_mag*premx*premy*premz2+8*Erem*p4y*p4z2*p4_mag*premx*premy*premz2+12*p4x3*p4y*premx2
        *premy*premz2+12*p4x*p4y*p4z2*premx2*premy*premz2+4*Erem*p4x*p4y2*p4_mag*premy2*premz2-4*Erem*p4x*p4z2*p4_mag*premy2*premz2
        +12*p4x2*p4y2*premx*premy2*premz2+4*p4x2*p4z2*premx*premy2*premz2+4*p4y2*p4z2*premx*premy2*premz2+4*p4z4*premx*premy2*premz2
        +4*p4x*p4y3*premy3*premz2+4*p4x*p4y*p4z2*premy3*premz2)

    # Defines domain_func which determins the domain of the two p2z functions.
    domain_func = (T**2-4*A*(Erem4*mwm2**2*p4x2+Erem6*p4x4+Erem4*mwm2**2
        *p4y2+Erem6*p4x2*p4y2+4*Erem4*p2z**2*p4x2*p4y2+4*Erem4*p2z**2*p4y4-4*Erem4*mwm2*p2z*p4x2*p4z-4*Erem4*mwm2*p2z*p4y2*p4z
        +Erem6*p4x2*p4z2+4*Erem4*p2z**2*p4x2*p4z2+Erem6*p4y2*p4z2+4*Erem4*p2z**2*p4y2*p4z2-2*Erem5*mwm2*p4x2*p4_mag-2*Erem5*mwm2
        *p4y2*p4_mag+4*Erem5*p2z*p4x2*p4z*p4_mag+4*Erem5*p2z*p4y2*p4z*p4_mag-4*Erem4*mwm2*p4x3*premx-4*Erem4*mwm2*p4x*p4y2*premx
        +8*Erem4*p2z*p4x3*p4z*premx+8*Erem4*p2z*p4x*p4y2*p4z*premx-4*Erem4*mwm2*p4x*p4z2*premx+8*Erem4*p2z*p4x*p4z3*premx+2*Erem3
        *mwm2**2*p4x*p4_mag*premx+2*Erem5*p4x3*p4_mag*premx+8*Erem3*p2z**2*p4x*p4y2*p4_mag*premx-8*Erem3*mwm2*p2z*p4x*p4z*p4_mag
        *premx+2*Erem5*p4x*p4z2*p4_mag*premx+8*Erem3*p2z**2*p4x*p4z2*p4_mag*premx+Erem2*mwm2**2*p4x2*premx2-Erem4*p4x4*premx2-2*Erem4
        *p4x2*p4y2*premx2+4*Erem2*p2z**2*p4x2*p4y2*premx2-4*Erem2*mwm2*p2z*p4x2*p4z*premx2+Erem2*mwm2**2*p4z2*premx2+4*Erem2*p2z
        **2*p4x2*p4z2*premx2-2*Erem4*p4y2*p4z2*premx2+4*Erem2*p2z**2*p4y2*p4z2*premx2-4*Erem2*mwm2*p2z*p4z3*premx2+Erem4*p4z4*premx2
        +4*Erem2*p2z**2*p4z4*premx2+2*Erem3*mwm2*p4y2*p4_mag*premx2-4*Erem3*p2z*p4y2*p4z*p4_mag*premx2-2*Erem3*mwm2*p4z2*p4_mag*premx2
        +4*Erem3*p2z*p4z3*p4_mag*premx2+4*Erem2*mwm2*p4x3*premx3+4*Erem2*mwm2*p4x*p4y2*premx3-8*Erem2*p2z*p4x3*p4z*premx3-8*Erem2
        *p2z*p4x*p4y2*p4z*premx3+4*Erem2*mwm2*p4x*p4z2*premx3-8*Erem2*p2z*p4x*p4z3*premx3-4*Erem3*p4x3*p4_mag*premx3-4*Erem3*p4x
        *p4z2*p4_mag*premx3-Erem2*p4x4*premx4+Erem2*p4x2*p4y2*premx4-3*Erem2*p4x2*p4z2*premx4+Erem2*p4y2*p4z2*premx4-2*Erem2*p4z4
        *premx4+2*Erem*mwm2*p4x2*p4_mag*premx4-4*Erem*p2z*p4x2*p4z*p4_mag*premx4+2*Erem*mwm2*p4z2*p4_mag*premx4-4*Erem*p2z*p4z3*p4_mag
        *premx4+2*Erem*p4x3*p4_mag*premx5+2*Erem*p4x*p4z2*p4_mag*premx5+p4x4*premx6+2*p4x2*p4z2*premx6+p4z4*premx6-2*Erem4*mwm2*p4x2
        *p4y*premy-2*Erem4*mwm2*p4y3*premy+4*Erem4*p2z*p4x2*p4y*p4z*premy+4*Erem4*p2z*p4y3*p4z*premy-4*Erem4*mwm2*p4y*p4z2*premy
        +8*Erem4*p2z*p4y*p4z3*premy+2*Erem3*mwm2**2*p4y*p4_mag*premy+2*Erem5*p4x2*p4y*p4_mag*premy-8*Erem3*p2z**2*p4x2*p4y*p4_mag
        *premy-8*Erem3*mwm2*p2z*p4y*p4z*p4_mag*premy+2*Erem5*p4y*p4z2*p4_mag*premy+8*Erem3*p2z**2*p4y*p4z2*p4_mag*premy+2*Erem2*mwm2
        **2*p4x*p4y*premx*premy+2*Erem4*p4x3*p4y*premx*premy-16*Erem2*p2z**2*p4x3*p4y*premx*premy-8*Erem2*p2z**2*p4x*p4y3*premx*premy
        -8*Erem2*mwm2*p2z*p4x*p4y*p4z*premx*premy+2*Erem4*p4x*p4y*p4z2*premx*premy-8*Erem2*p2z**2*p4x*p4y*p4z2*premx*premy+4*Erem2
        *mwm2*p4x2*p4y*premx2*premy+2*Erem2*mwm2*p4y3*premx2*premy-8*Erem2*p2z*p4x2*p4y*p4z*premx2*premy-4*Erem2*p2z*p4y3*p4z*premx2
        *premy+6*Erem2*mwm2*p4y*p4z2*premx2*premy-12*Erem2*p2z*p4y*p4z3*premx2*premy-4*Erem3*p4x2*p4y*p4_mag*premx2*premy-8*Erem
        *p2z**2*p4x2*p4y*p4_mag*premx2*premy-4*Erem3*p4y*p4z2*p4_mag*premx2*premy-8*Erem*p2z**2*p4y*p4z2*p4_mag*premx2*premy-4*Erem2
        *p4x3*p4y*premx3*premy-4*Erem2*p4x*p4y*p4z2*premx3*premy-2*mwm2*p4x2*p4y*premx4*premy+4*p2z*p4x2*p4y*p4z*premx4*premy-2*mwm2
        *p4y*p4z2*premx4*premy+4*p2z*p4y*p4z3*premx4*premy+2*Erem*p4x2*p4y*p4_mag*premx4*premy+2*Erem*p4y*p4z2*p4_mag*premx4*premy
        +2*p4x3*p4y*premx5*premy+2*p4x*p4y*p4z2*premx5*premy-Erem2*mwm2**2*p4x2*premy2-2*Erem4*p4x4*premy2+4*Erem2*p2z**2*p4x4*premy2
        -Erem4*p4x2*p4y2*premy2-8*Erem2*p2z**2*p4x2*p4y2*premy2-8*Erem2*p2z**2*p4y4*premy2+4*Erem2*mwm2*p2z*p4x2*p4z*premy2+Erem2
        *mwm2**2*p4z2*premy2-Erem4*p4x2*p4z2*premy2-Erem4*p4y2*p4z2*premy2-8*Erem2*p2z**2*p4y2*p4z2*premy2-4*Erem2*mwm2*p2z*p4z3
        *premy2+Erem4*p4z4*premy2+4*Erem2*p2z**2*p4z4*premy2+2*Erem3*mwm2*p4x2*p4_mag*premy2+4*Erem3*mwm2*p4y2*p4_mag*premy2-4*Erem3
        *p2z*p4x2*p4z*p4_mag*premy2-8*Erem3*p2z*p4y2*p4z*p4_mag*premy2-2*Erem3*mwm2*p4z2*p4_mag*premy2+4*Erem3*p2z*p4z3*p4_mag*premy2
        +4*Erem2*mwm2*p4x3*premx*premy2+8*Erem2*mwm2*p4x*p4y2*premx*premy2-8*Erem2*p2z*p4x3*p4z*premx*premy2-16*Erem2*p2z*p4x*p4y2
        *p4z*premx*premy2+4*Erem2*mwm2*p4x*p4z2*premx*premy2-8*Erem2*p2z*p4x*p4z3*premx*premy2-2*Erem*mwm2**2*p4x*p4_mag*premx*premy2
        -4*Erem3*p4x3*p4_mag*premx*premy2+8*Erem*p2z**2*p4x3*p4_mag*premx*premy2-8*Erem*p2z**2*p4x*p4y2*p4_mag*premx*premy2+8*Erem
        *mwm2*p2z*p4x*p4z*p4_mag*premx*premy2-4*Erem3*p4x*p4z2*p4_mag*premx*premy2-mwm2**2*p4x2*premx2*premy2+4*p2z**2*p4x4*premx2
        *premy2+4*p2z**2*p4x2*p4y2*premx2*premy2+4*mwm2*p2z*p4x2*p4z*premx2*premy2-mwm2**2*p4z2*premx2*premy2-4*Erem2*p4x2*p4z2*premx2
        *premy2+4*p2z**2*p4x2*p4z2*premx2*premy2+4*p2z**2*p4y2*p4z2*premx2*premy2+4*mwm2*p2z*p4z3*premx2*premy2-4*Erem2*p4z4*premx2
        *premy2+2*Erem*mwm2*p4x2*p4_mag*premx2*premy2-2*Erem*mwm2*p4y2*p4_mag*premx2*premy2-4*Erem*p2z*p4x2*p4z*p4_mag*premx2*premy2
        +4*Erem*p2z*p4y2*p4z*p4_mag*premx2*premy2+4*Erem*mwm2*p4z2*p4_mag*premx2*premy2-8*Erem*p2z*p4z3*p4_mag*premx2*premy2-4*mwm2
        *p4x*p4y2*premx3*premy2+8*p2z*p4x*p4y2*p4z*premx3*premy2+4*Erem*p4x3*p4_mag*premx3*premy2+4*Erem*p4x*p4z2*p4_mag*premx3*premy2
        +2*p4x4*premx4*premy2+p4x2*p4y2*premx4*premy2+5*p4x2*p4z2*premx4*premy2+p4y2*p4z2*premx4*premy2+3*p4z4*premx4*premy2+2*Erem2
        *mwm2*p4x2*p4y*premy3+4*Erem2*mwm2*p4y3*premy3-4*Erem2*p2z*p4x2*p4y*p4z*premy3-8*Erem2*p2z*p4y3*p4z*premy3+6*Erem2*mwm2*p4y
        *p4z2*premy3-12*Erem2*p2z*p4y*p4z3*premy3-2*Erem*mwm2**2*p4y*p4_mag*premy3-4*Erem3*p4x2*p4y*p4_mag*premy3+8*Erem*p2z**2*p4x2
        *p4y*p4_mag*premy3+8*Erem*mwm2*p2z*p4y*p4z*p4_mag*premy3-4*Erem3*p4y*p4z2*p4_mag*premy3-8*Erem*p2z**2*p4y*p4z2*p4_mag*premy3
        -2*mwm2**2*p4x*p4y*premx*premy3-4*Erem2*p4x3*p4y*premx*premy3+8*p2z**2*p4x3*p4y*premx*premy3+8*p2z**2*p4x*p4y3*premx*premy3
        +8*mwm2*p2z*p4x*p4y*p4z*premx*premy3-4*Erem2*p4x*p4y*p4z2*premx*premy3-2*mwm2*p4x2*p4y*premx2*premy3-2*mwm2*p4y3*premx2*premy3
        +4*p2z*p4x2*p4y*p4z*premx2*premy3+4*p2z*p4y3*p4z*premx2*premy3-4*mwm2*p4y*p4z2*premx2*premy3+8*p2z*p4y*p4z3*premx2*premy3
        +4*Erem*p4x2*p4y*p4_mag*premx2*premy3+4*Erem*p4y*p4z2*p4_mag*premx2*premy3+4*p4x3*p4y*premx3*premy3+4*p4x*p4y*p4z2*premx3
        *premy3+Erem2*p4x4*premy4-mwm2**2*p4y2*premy4-Erem2*p4x2*p4y2*premy4+4*p2z**2*p4x2*p4y2*premy4+4*p2z**2*p4y4*premy4+4*mwm2
        *p2z*p4y2*p4z*premy4-mwm2**2*p4z2*premy4-Erem2*p4x2*p4z2*premy4+4*p2z**2*p4x2*p4z2*premy4-Erem2*p4y2*p4z2*premy4+4*p2z**2
        *p4y2*p4z2*premy4+4*mwm2*p2z*p4z3*premy4-2*Erem2*p4z4*premy4-2*Erem*mwm2*p4y2*p4_mag*premy4+4*Erem*p2z*p4y2*p4z*p4_mag*premy4
        +2*Erem*mwm2*p4z2*p4_mag*premy4-4*Erem*p2z*p4z3*p4_mag*premy4-4*mwm2*p4x*p4y2*premx*premy4+8*p2z*p4x*p4y2*p4z*premx*premy4
        +2*Erem*p4x3*p4_mag*premx*premy4+2*Erem*p4x*p4z2*p4_mag*premx*premy4+p4x4*premx2*premy4+2*p4x2*p4y2*premx2*premy4+4*p4x2
        *p4z2*premx2*premy4+2*p4y2*p4z2*premx2*premy4+3*p4z4*premx2*premy4-2*mwm2*p4y3*premy5+4*p2z*p4y3*p4z*premy5-2*mwm2*p4y*p4z2
        *premy5+4*p2z*p4y*p4z3*premy5+2*Erem*p4x2*p4y*p4_mag*premy5+2*Erem*p4y*p4z2*p4_mag*premy5+2*p4x3*p4y*premx*premy5+2*p4x*p4y
        *p4z2*premx*premy5+p4x2*p4y2*premy6+p4x2*p4z2*premy6+p4y2*p4z2*premy6+p4z4*premy6-4*Erem4*p2z*p4x4*premz-12*Erem4*p2z*p4x2
        *p4y2*premz-8*Erem4*p2z*p4y4*premz+4*Erem4*mwm2*p4x2*p4z*premz+4*Erem4*mwm2*p4y2*p4z*premz-12*Erem4*p2z*p4x2*p4z2*premz-12
        *Erem4*p2z*p4y2*p4z2*premz+4*Erem3*mwm2*p2z*p4x2*p4_mag*premz+4*Erem3*mwm2*p2z*p4y2*p4_mag*premz-4*Erem5*p4x2*p4z*p4_mag
        *premz-8*Erem3*p2z**2*p4x2*p4z*p4_mag*premz-4*Erem5*p4y2*p4z*p4_mag*premz-8*Erem3*p2z**2*p4y2*p4z*p4_mag*premz+8*Erem2*mwm2
        *p2z*p4x3*premx*premz+8*Erem2*mwm2*p2z*p4x*p4y2*premx*premz-8*Erem4*p4x3*p4z*premx*premz-16*Erem2*p2z**2*p4x3*p4z*premx*premz
        -8*Erem4*p4x*p4y2*p4z*premx*premz-16*Erem2*p2z**2*p4x*p4y2*p4z*premx*premz+8*Erem2*mwm2*p2z*p4x*p4z2*premx*premz-8*Erem4
        *p4x*p4z3*premx*premz-16*Erem2*p2z**2*p4x*p4z3*premx*premz-8*Erem3*p2z*p4x3*p4_mag*premx*premz-16*Erem3*p2z*p4x*p4y2*p4_mag
        *premx*premz+8*Erem3*mwm2*p4x*p4z*p4_mag*premx*premz-24*Erem3*p2z*p4x*p4z2*p4_mag*premx*premz-4*Erem2*p2z*p4x2*p4y2*premx2
        *premz+4*Erem2*mwm2*p4x2*p4z*premx2*premz-12*Erem2*p2z*p4x2*p4z2*premx2*premz-4*Erem2*p2z*p4y2*p4z2*premx2*premz+4*Erem2
        *mwm2*p4z3*premx2*premz-12*Erem2*p2z*p4z4*premx2*premz+4*Erem*mwm2*p2z*p4x2*p4_mag*premx2*premz-8*Erem*p2z**2*p4x2*p4z*p4_mag
        *premx2*premz+4*Erem3*p4y2*p4z*p4_mag*premx2*premz+4*Erem*mwm2*p2z*p4z2*p4_mag*premx2*premz-4*Erem3*p4z3*p4_mag*premx2*premz
        -8*Erem*p2z**2*p4z3*p4_mag*premx2*premz+8*Erem2*p4x3*p4z*premx3*premz+8*Erem2*p4x*p4y2*p4z*premx3*premz+8*Erem2*p4x*p4z3
        *premx3*premz+8*Erem*p2z*p4x3*p4_mag*premx3*premz+8*Erem*p2z*p4x*p4z2*p4_mag*premx3*premz+4*p2z*p4x4*premx4*premz+8*p2z*p4x2
        *p4z2*premx4*premz+4*p2z*p4z4*premx4*premz+4*Erem*p4x2*p4z*p4_mag*premx4*premz+4*Erem*p4z3*p4_mag*premx4*premz+4*Erem2*mwm2
        *p2z*p4x2*p4y*premy*premz+4*Erem2*mwm2*p2z*p4y3*premy*premz-4*Erem4*p4x2*p4y*p4z*premy*premz-8*Erem2*p2z**2*p4x2*p4y*p4z
        *premy*premz-4*Erem4*p4y3*p4z*premy*premz-8*Erem2*p2z**2*p4y3*p4z*premy*premz+8*Erem2*mwm2*p2z*p4y*p4z2*premy*premz-8*Erem4
        *p4y*p4z3*premy*premz-16*Erem2*p2z**2*p4y*p4z3*premy*premz+8*Erem3*p2z*p4x2*p4y*p4_mag*premy*premz+8*Erem3*mwm2*p4y*p4z*p4_mag
        *premy*premz-24*Erem3*p2z*p4y*p4z2*p4_mag*premy*premz+24*Erem2*p2z*p4x3*p4y*premx*premy*premz+16*Erem2*p2z*p4x*p4y3*premx
        *premy*premz+8*Erem2*mwm2*p4x*p4y*p4z*premx*premy*premz+8*Erem2*p2z*p4x*p4y*p4z2*premx*premy*premz-4*mwm2*p2z*p4x2*p4y*premx2
        *premy*premz+8*Erem2*p4x2*p4y*p4z*premx2*premy*premz+8*p2z**2*p4x2*p4y*p4z*premx2*premy*premz+4*Erem2*p4y3*p4z*premx2*premy
        *premz-4*mwm2*p2z*p4y*p4z2*premx2*premy*premz+12*Erem2*p4y*p4z3*premx2*premy*premz+8*p2z**2*p4y*p4z3*premx2*premy*premz+24
        *Erem*p2z*p4x2*p4y*p4_mag*premx2*premy*premz+24*Erem*p2z*p4y*p4z2*p4_mag*premx2*premy*premz+8*p2z*p4x3*p4y*premx3*premy*premz
        +8*p2z*p4x*p4y*p4z2*premx3*premy*premz-4*p4x2*p4y*p4z*premx4*premy*premz-4*p4y*p4z3*premx4*premy*premz-4*Erem2*p2z*p4x4*premy2
        *premz+16*Erem2*p2z*p4x2*p4y2*premy2*premz+16*Erem2*p2z*p4y4*premy2*premz-4*Erem2*mwm2*p4x2*p4z*premy2*premz+16*Erem2*p2z
        *p4y2*p4z2*premy2*premz+4*Erem2*mwm2*p4z3*premy2*premz-12*Erem2*p2z*p4z4*premy2*premz-4*Erem*mwm2*p2z*p4y2*p4_mag*premy2
        *premz+4*Erem3*p4x2*p4z*p4_mag*premy2*premz+8*Erem3*p4y2*p4z*p4_mag*premy2*premz+8*Erem*p2z**2*p4y2*p4z*p4_mag*premy2*premz
        +4*Erem*mwm2*p2z*p4z2*p4_mag*premy2*premz-4*Erem3*p4z3*p4_mag*premy2*premz-8*Erem*p2z**2*p4z3*p4_mag*premy2*premz-8*mwm2
        *p2z*p4x*p4y2*premx*premy2*premz+8*Erem2*p4x3*p4z*premx*premy2*premz+16*Erem2*p4x*p4y2*p4z*premx*premy2*premz+16*p2z**2*p4x
        *p4y2*p4z*premx*premy2*premz+8*Erem2*p4x*p4z3*premx*premy2*premz-8*Erem*p2z*p4x3*p4_mag*premx*premy2*premz+16*Erem*p2z*p4x
        *p4y2*p4_mag*premx*premy2*premz-8*Erem*mwm2*p4x*p4z*p4_mag*premx*premy2*premz+8*Erem*p2z*p4x*p4z2*p4_mag*premx*premy2*premz
        -4*p2z*p4x4*premx2*premy2*premz-4*p2z*p4x2*p4y2*premx2*premy2*premz-4*mwm2*p4x2*p4z*premx2*premy2*premz+4*p2z*p4x2*p4z2*premx2
        *premy2*premz-4*p2z*p4y2*p4z2*premx2*premy2*premz-4*mwm2*p4z3*premx2*premy2*premz+8*p2z*p4z4*premx2*premy2*premz+4*Erem*p4x2
        *p4z*p4_mag*premx2*premy2*premz-4*Erem*p4y2*p4z*p4_mag*premx2*premy2*premz+8*Erem*p4z3*p4_mag*premx2*premy2*premz-8*p4x*p4y2
        *p4z*premx3*premy2*premz-4*mwm2*p2z*p4y3*premy3*premz+4*Erem2*p4x2*p4y*p4z*premy3*premz+8*Erem2*p4y3*p4z*premy3*premz+8*p2z
        **2*p4y3*p4z*premy3*premz-4*mwm2*p2z*p4y*p4z2*premy3*premz+12*Erem2*p4y*p4z3*premy3*premz+8*p2z**2*p4y*p4z3*premy3*premz
        -8*Erem*p2z*p4x2*p4y*p4_mag*premy3*premz-8*Erem*mwm2*p4y*p4z*p4_mag*premy3*premz+24*Erem*p2z*p4y*p4z2*p4_mag*premy3*premz
        -8*p2z*p4x3*p4y*premx*premy3*premz-16*p2z*p4x*p4y3*premx*premy3*premz-8*mwm2*p4x*p4y*p4z*premx*premy3*premz+8*p2z*p4x*p4y
        *p4z2*premx*premy3*premz-4*p4x2*p4y*p4z*premx2*premy3*premz-4*p4y3*p4z*premx2*premy3*premz-8*p4y*p4z3*premx2*premy3*premz
        -4*p2z*p4x2*p4y2*premy4*premz-8*p2z*p4y4*premy4*premz-4*mwm2*p4y2*p4z*premy4*premz-4*p2z*p4x2*p4z2*premy4*premz-4*p2z*p4y2
        *p4z2*premy4*premz-4*mwm2*p4z3*premy4*premz+4*p2z*p4z4*premy4*premz-4*Erem*p4y2*p4z*p4_mag*premy4*premz+4*Erem*p4z3*p4_mag
        *premy4*premz-8*p4x*p4y2*p4z*premx*premy4*premz-4*p4y3*p4z*premy5*premz-4*p4y*p4z3*premy5*premz+2*Erem4*p4x4*premz2+4*Erem2
        *p2z**2*p4x4*premz2+6*Erem4*p4x2*p4y2*premz2+4*Erem2*p2z**2*p4x2*p4y2*premz2+4*Erem4*p4y4*premz2+6*Erem4*p4x2*p4z2*premz2
        +4*Erem2*p2z**2*p4x2*p4z2*premz2+6*Erem4*p4y2*p4z2*premz2+4*Erem2*p2z**2*p4y2*p4z2*premz2-2*Erem3*mwm2*p4x2*p4_mag*premz2
        -2*Erem3*mwm2*p4y2*p4_mag*premz2+12*Erem3*p2z*p4x2*p4z*p4_mag*premz2+12*Erem3*p2z*p4y2*p4z*p4_mag*premz2-4*Erem2*mwm2*p4x3
        *premx*premz2-4*Erem2*mwm2*p4x*p4y2*premx*premz2+24*Erem2*p2z*p4x3*p4z*premx*premz2+24*Erem2*p2z*p4x*p4y2*p4z*premx*premz2
        -4*Erem2*mwm2*p4x*p4z2*premx*premz2+24*Erem2*p2z*p4x*p4z3*premx*premz2+4*Erem3*p4x3*p4_mag*premx*premz2+8*Erem*p2z**2*p4x3
        *p4_mag*premx*premz2+8*Erem3*p4x*p4y2*p4_mag*premx*premz2+12*Erem3*p4x*p4z2*p4_mag*premx*premz2+8*Erem*p2z**2*p4x*p4z2*p4_mag
        *premx*premz2+4*p2z**2*p4x4*premx2*premz2+2*Erem2*p4x2*p4y2*premx2*premz2+6*Erem2*p4x2*p4z2*premx2*premz2+8*p2z**2*p4x2*p4z2
        *premx2*premz2+2*Erem2*p4y2*p4z2*premx2*premz2+6*Erem2*p4z4*premx2*premz2+4*p2z**2*p4z4*premx2*premz2-2*Erem*mwm2*p4x2*p4_mag
        *premx2*premz2+12*Erem*p2z*p4x2*p4z*p4_mag*premx2*premz2-2*Erem*mwm2*p4z2*p4_mag*premx2*premz2+12*Erem*p2z*p4z3*p4_mag*premx2
        *premz2-4*Erem*p4x3*p4_mag*premx3*premz2-4*Erem*p4x*p4z2*p4_mag*premx3*premz2-2*p4x4*premx4*premz2-4*p4x2*p4z2*premx4*premz2
        -2*p4z4*premx4*premz2-2*Erem2*mwm2*p4x2*p4y*premy*premz2-2*Erem2*mwm2*p4y3*premy*premz2+12*Erem2*p2z*p4x2*p4y*p4z*premy*premz2
        +12*Erem2*p2z*p4y3*p4z*premy*premz2-4*Erem2*mwm2*p4y*p4z2*premy*premz2+24*Erem2*p2z*p4y*p4z3*premy*premz2-4*Erem3*p4x2*p4y
        *p4_mag*premy*premz2+8*Erem*p2z**2*p4x2*p4y*p4_mag*premy*premz2+12*Erem3*p4y*p4z2*p4_mag*premy*premz2+8*Erem*p2z**2*p4y*p4z2
        *p4_mag*premy*premz2-12*Erem2*p4x3*p4y*premx*premy*premz2+8*p2z**2*p4x3*p4y*premx*premy*premz2-8*Erem2*p4x*p4y3*premx*premy
        *premz2-4*Erem2*p4x*p4y*p4z2*premx*premy*premz2+8*p2z**2*p4x*p4y*p4z2*premx*premy*premz2+2*mwm2*p4x2*p4y*premx2*premy*premz2
        -12*p2z*p4x2*p4y*p4z*premx2*premy*premz2+2*mwm2*p4y*p4z2*premx2*premy*premz2-12*p2z*p4y*p4z3*premx2*premy*premz2-12*Erem
        *p4x2*p4y*p4_mag*premx2*premy*premz2-12*Erem*p4y*p4z2*p4_mag*premx2*premy*premz2-4*p4x3*p4y*premx3*premy*premz2-4*p4x*p4y
        *p4z2*premx3*premy*premz2+2*Erem2*p4x4*premy2*premz2-8*Erem2*p4x2*p4y2*premy2*premz2+4*p2z**2*p4x2*p4y2*premy2*premz2-8*Erem2
        *p4y4*premy2*premz2+4*p2z**2*p4x2*p4z2*premy2*premz2-8*Erem2*p4y2*p4z2*premy2*premz2+4*p2z**2*p4y2*p4z2*premy2*premz2+6*Erem2
        *p4z4*premy2*premz2+4*p2z**2*p4z4*premy2*premz2+2*Erem*mwm2*p4y2*p4_mag*premy2*premz2-12*Erem*p2z*p4y2*p4z*p4_mag*premy2
        *premz2-2*Erem*mwm2*p4z2*p4_mag*premy2*premz2+12*Erem*p2z*p4z3*p4_mag*premy2*premz2+4*mwm2*p4x*p4y2*premx*premy2*premz2-24
        *p2z*p4x*p4y2*p4z*premx*premy2*premz2+4*Erem*p4x3*p4_mag*premx*premy2*premz2-8*Erem*p4x*p4y2*p4_mag*premx*premy2*premz2-4
        *Erem*p4x*p4z2*p4_mag*premx*premy2*premz2+2*p4x4*premx2*premy2*premz2+2*p4x2*p4y2*premx2*premy2*premz2-2*p4x2*p4z2*premx2
        *premy2*premz2+2*p4y2*p4z2*premx2*premy2*premz2-4*p4z4*premx2*premy2*premz2+2*mwm2*p4y3*premy3*premz2-12*p2z*p4y3*p4z*premy3
        *premz2+2*mwm2*p4y*p4z2*premy3*premz2-12*p2z*p4y*p4z3*premy3*premz2+4*Erem*p4x2*p4y*p4_mag*premy3*premz2-12*Erem*p4y*p4z2
        *p4_mag*premy3*premz2+4*p4x3*p4y*premx*premy3*premz2+8*p4x*p4y3*premx*premy3*premz2-4*p4x*p4y*p4z2*premx*premy3*premz2+2
        *p4x2*p4y2*premy4*premz2+4*p4y4*premy4*premz2+2*p4x2*p4z2*premy4*premz2+2*p4y2*p4z2*premy4*premz2-2*p4z4*premy4*premz2-4
        *Erem2*p2z*p4x4*premz3-4*Erem2*p2z*p4x2*p4y2*premz3-4*Erem2*p2z*p4x2*p4z2*premz3-4*Erem2*p2z*p4y2*p4z2*premz3-4*Erem3*p4x2
        *p4z*p4_mag*premz3-4*Erem3*p4y2*p4z*p4_mag*premz3-8*Erem2*p4x3*p4z*premx*premz3-8*Erem2*p4x*p4y2*p4z*premx*premz3-8*Erem2
        *p4x*p4z3*premx*premz3-8*Erem*p2z*p4x3*p4_mag*premx*premz3-8*Erem*p2z*p4x*p4z2*p4_mag*premx*premz3-4*p2z*p4x4*premx2*premz3
        -8*p2z*p4x2*p4z2*premx2*premz3-4*p2z*p4z4*premx2*premz3-4*Erem*p4x2*p4z*p4_mag*premx2*premz3-4*Erem*p4z3*p4_mag*premx2*premz3
        -4*Erem2*p4x2*p4y*p4z*premy*premz3-4*Erem2*p4y3*p4z*premy*premz3-8*Erem2*p4y*p4z3*premy*premz3-8*Erem*p2z*p4x2*p4y*p4_mag
        *premy*premz3-8*Erem*p2z*p4y*p4z2*p4_mag*premy*premz3-8*p2z*p4x3*p4y*premx*premy*premz3-8*p2z*p4x*p4y*p4z2*premx*premy*premz3
        +4*p4x2*p4y*p4z*premx2*premy*premz3+4*p4y*p4z3*premx2*premy*premz3-4*p2z*p4x2*p4y2*premy2*premz3-4*p2z*p4x2*p4z2*premy2*premz3
        -4*p2z*p4y2*p4z2*premy2*premz3-4*p2z*p4z4*premy2*premz3+4*Erem*p4y2*p4z*p4_mag*premy2*premz3-4*Erem*p4z3*p4_mag*premy2*premz3
        +8*p4x*p4y2*p4z*premx*premy2*premz3+4*p4y3*p4z*premy3*premz3+4*p4y*p4z3*premy3*premz3+Erem2*p4x4*premz4+Erem2*p4x2*p4y2*premz4
        +Erem2*p4x2*p4z2*premz4+Erem2*p4y2*p4z2*premz4+2*Erem*p4x3*p4_mag*premx*premz4+2*Erem*p4x*p4z2*p4_mag*premx*premz4+p4x4*premx2
        *premz4+2*p4x2*p4z2*premx2*premz4+p4z4*premx2*premz4+2*Erem*p4x2*p4y*p4_mag*premy*premz4+2*Erem*p4y*p4z2*p4_mag*premy*premz4
        +2*p4x3*p4y*premx*premy*premz4+2*p4x*p4y*p4z2*premx*premy*premz4+p4x2*p4y2*premy2*premz4+p4x2*p4z2*premy2*premz4+p4y2*p4z2
        *premy2*premz4+p4z4*premy2*premz4))

    sqrt_domain = sqrt(domain_func)

    B = -T-sqrt_domain

    C = (-(Erem2*mwm2*p4y)+2*Erem2*p2z*p4y*p4z+Erem3*p4y*p4_mag-Erem*p4y*p4_mag*premx2+Erem2*p4x2*premy+Erem2*p4y2*premy+Erem2*p4z2
        *premy-Erem*mwm2*p4_mag*premy+2*Erem*p2z*p4z*p4_mag*premy-p4x2*premx2*premy-p4y2*premx2*premy-p4z2*premx2*premy-Erem*p4y
        *p4_mag*premy2-p4x2*premy3-p4y2*premy3-p4z2*premy3-2*Erem2*p4y*p4z*premz-2*Erem*p2z*p4y*p4_mag*premz-2*p2z*p4x2*premy*premz
        -2*p2z*p4y2*premy*premz-2*p2z*p4z2*premy*premz-2*Erem*p4z*p4_mag*premy*premz+Erem*p4y*p4_mag*premz2+p4x2*premy*premz2+p4y2
        *premy*premz2+p4z2*premy*premz2-(Erem2*p4x*p4y*B)/A+(Erem*p4y*p4_mag*premx*B)/A-(Erem*p4x*p4_mag*premy*B)/A
        +(p4x2*premx*premy*B)/A+(p4y2*premx*premy*B)/A+(p4z2*premx*premy*B)/A)

    D = (Erem2*p4y2-p4x2*premy2-p4y2*premy2-p4z2*premy2)

    H = (-T+sqrt_domain)          # = B+2*sqrt_domain

    K = (-(Erem2*mwm2*p4y)+2*Erem2*p2z*p4y*p4z+Erem3*p4y*p4_mag-Erem*p4y*p4_mag*premx2+Erem2*p4x2*premy+Erem2*p4y2*premy+Erem2*p4z2
        *premy-Erem*mwm2*p4_mag*premy+2*Erem*p2z*p4z*p4_mag*premy-p4x2*premx2*premy-p4y2*premx2*premy-p4z2*premx2*premy-Erem*p4y
        *p4_mag*premy2-p4x2*premy3-p4y2*premy3-p4z2*premy3-2*Erem2*p4y*p4z*premz-2*Erem*p2z*p4y*p4_mag*premz-2*p2z*p4x2*premy*premz
        -2*p2z*p4y2*premy*premz-2*p2z*p4z2*premy*premz-2*Erem*p4z*p4_mag*premy*premz+Erem*p4y*p4_mag*premz2+p4x2*premy*premz2+p4y2
        *premy*premz2+p4z2*premy*premz2-(Erem2*p4x*p4y*H)/A+(Erem*p4y*p4_mag*premx*H)/A-(Erem*p4x*p4_mag*premy*H)/A+(p4x2*premx*premy
        *H)/A+(p4y2*premx*premy*H)/A+(p4z2*premx*premy*H)/A)

    # Defines functions for neutrino x.
    p2x_func1 = premx-B/(2*A)
    p2x_func2 = premx-H/(2*A)

    # Defines functions for neutrino y.
    p2y_func1 = premy-C/(2*D)
    p2y_func2 = premy-K/(2*D)

    # Define p2z_func1.
    p2z_func1 = ((sqrt(p1x**2 + p1y**2 + p1z**2) + sqrt(p2x_func1**2 + p2y_func1**2 + p2z**2))**2
                 -(p1x+p2x_func1)**2 - (p1y+p2y_func1)**2 - (p1z+p2z)**2 - mwm2)
    # Define p2z_func2.
    p2z_func2 = ((sqrt(p1x**2 + p1y**2 + p1z**2) + sqrt(p2x_func2**2 + p2y_func2**2 + p2z**2))**2
                 -(p1x+p2x_func2)**2 - (p1y+p2y_func2)**2 - (p1z+p2z)**2 - mwm2)

    # Define p5x_funcs.
    p5x_func1 = premx-p2x_func1               # = B/(2*A)
    p5x_func2 = premx-p2x_func2               # = H/(2*A)
    
    # Define p5y_funcs.
    p5y_func1 = premy-p2y_func1               # = C/D
    p5y_func2 = premy-p2y_func2               # = K/D
    
    # Define p5z_funcs.
    p5z_func1 = -p2z+premz
    p5z_func2 = -p2z+premz

    # Simplify funcs.
    domain_func = domain_func.expand()
    p2x_func1 = p2x_func1.expand()
    p2x_func2 = p2x_func2.expand()
    p2y_func1 = p2y_func1.expand()
    p2y_func2 = p2y_func2.expand()
    p2z_func1 = p2z_func1.expand()
    p2z_func2 = p2z_func2.expand()
    p5x_func1 = p5x_func1.expand()
    p5x_func2 = p5x_func2.expand()
    p5y_func1 = p5y_func1.expand()
    p5y_func2 = p5y_func2.expand()
    p5z_func1 = p5z_func1.expand()
    p5z_func2 = p5z_func2.expand()

    func1s = (p2x_func1, p2y_func1, p2z_func1, p5x_func1, p5y_func1, p5z_func1)
    func2s = (p2x_func2, p2y_func2, p2z_func2, p5x_func2, p5y_func2, p5z_func2)
    
    return domain_func, func1s, func2s  