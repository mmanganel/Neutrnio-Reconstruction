# Neurecon

Neurecon is a Python library that provides a method for numerically reconstructing the momenta of neutrinos in electron-positron particle colliders. For a top/anti-top dilepton event, given the measured x, y, and z momenta components of the muons and bottom quarks, the method will return the full kinetic reconstruction of the neutrino and anti-neutrino. 


## Requirements
* [numpy](http://www.numpy.org/)
* [sympy](https://www.sympy.org/en/index.html)

## Files
* reconstruction: 
Contains the *reconstruct* method which takes the measured particles' momenta, along with the center of mass energy and constained mass of the W boson, as input and produces the set of solutions for the reconstructed neutrino and anti-neutrino momenta. 
* interpolation: 
A module for basic polynomial interpolation. Used by reconstruct method.

## Reconstruct
*reconstruct(edata, mwm=80.4, cme=1000)*: Reconstructs the momentum of the neutrino and anti-neutrino, given the momentum of the muons and bottom quarks. 

INPUT:

* edata: An iterable containing the x, y, and z momentum in GeV of the charged leptons and bottom quarks, in the following order,
      
      edata := [amux, amuy, amuz, b1x, b1y, b1z, mux, muy, muz, b2x, b2y, b2z],
      with notation:
      amu := anti-muon
      b1 := bottom quark 1*
      mu := muon
      b2 := bottom quark 2*
      * The charge of the bottom quark is assumed to be unknown.

* mwm(default=80.4): The constrained mass of the W boson in GeV.

* cme(default=1000): The center of mass energy.
        
OUTPUT:

* solutions: A list of the reconstructed neutrino and anti-neutrino x, y, and z-momenta as a tuple, for each possible solution of nuz, e.g. [(nux, nuy, nuz, anux, anuy, anuz), ...].


## Usage

```python
from neurecon import reconstruct

# Measured x, y, z momenta of the anti-muon, first bottom quark, muon, and second bottom quark in GeV.
ev = [-125.91139241, -48.98908297, -78.183743342, 
      -123.8118607, 53.940269187, -78.400421564,
      103.09571851, -6.5923468262, 68.47397934, 
      37.24678799, -39.496514832, 38.960820109]
      
center_mass_eng = 1000    # Gev
w_mass = 80.4

solutions = reconstruct(ev, mwm=w_mass, cme=center_mass_eng)

print('Number of solutions:', len(solutions))
for i, sol in enumerate(solutions):
    print(i+1, sol)      # (nux, nuy, nuz, anux, anuy, anuz)
```
Output:
```
Number of solutions: 2
1 (-93.46454134528696, -50.83231089912927, -159.54421086481028, 202.84528795528695, 91.96998634032927, 208.6935763218103)
2 (-184.7487135537596, -18.089040277773734, -26.6171658538459, 294.1294601637596, 59.226715718973736, 75.76653131084592)
```


## License
[MIT](https://choosealicense.com/licenses/mit/)
