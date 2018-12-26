# neurecon

neurecon is a Python library that provides a method for numerically reconstructing the momenta of neutrnios in electron-positron particle colliders. Given the measured x, y, and z momentum components of the meuons and bottom quarks, the method will retern the reconstructed momenta of the neutrino and anti-neutrino. 

## Requirements
* numpy
* sympy

## Files
* reconstruction
* interpolation
Includes implementation of basic polynomial interpolation


## Usage

```python
from neurecon.reconstruction import reconstruct

e1 = [-125.91139241, -48.98908297, -78.183743342, 
      -123.8118607, 53.940269187, -78.400421564,
      103.09571851, -6.5923468262, 68.47397934, 
      37.24678799, -39.496514832, 38.960820109]
      
center_mass_eng = 1000    # Gev
w_mass = 80.4

solutions, rcoeffs = reconstruct(e1, w_mass, center_mass_eng)

print('Number of solutions:', len(solutions))
for i, sol in enumerate(solutions):
    print(i+1, sol)
print('r-squared coeffs:', rcoeffs)
```
Output:
```
Number of solutions: 2
1 (-93.46454134528696, -50.83231089912927, -159.54421086481028, 202.84528795528695, 91.96998634032927, 208.6935763218103)
2 (-184.7487135537596, -18.089040277773734, -26.6171658538459, 294.1294601637596, 59.226715718973736, 75.76653131084592)
r-squared coeffs: (0.9803735804321968, 0.9822214569539975)
```


## License
[MIT](https://choosealicense.com/licenses/mit/)
