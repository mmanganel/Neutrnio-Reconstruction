# neurecon

neurecon is a Python library that provides a method for numerically reconstructing the momenta of neutrnios in electron-positron particle colliders. Given the measured x, y, and z momentum components of the meuons and bottom quarks, the method will retern the reconstructed momenta of the neutrino and anti-neutrino. 

## Requirements
*numpy
*sympy

## Files
*reconstruction
*interpolation
Includes implementation of basic polynomial interpolation


## Usage

```python
from neurecon.reconstruction import reconstruct

e1 = [-125.91139241, -48.98908297, -78.183743342, 
      -123.8118607, 53.940269187, -78.400421564,
      103.09571851, -6.5923468262, 68.47397934, 
      37.24678799, -39.496514832, 38.960820109]

solutions, rcoeffs = reconstruct(e1, 80.4, 1000)

print('Number of solutions:', len(solutions))
for i, sol in enumerate(solutions):
    print(i+1, sol)
print('r-squared coeffs:', rcoeffs)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
