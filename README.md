# Raytracing

This is a simple raytrace module allowing to compute the emission from an astrophysical gas.

## Description

The user defines the spatial distribution of the gas and its velocity field. The user also needs to define the fractionl population of the energy levels, either directly, or via a temperature. The code then raytraces the model, taking into account optical depth and the velocity field. A typical application case is the raytracing of a Keplerian disk. The main output is a 3D emission cube (2 spatial dimensions and one spectral dimension), which can be compared to observations. The atomic data needed for the calculation are read from a [LAMDA](https://home.strw.leidenuniv.nl/~moldata/) data file.

## Getting Started

### Dependencies

* numpy
* scipy
* matplotlib
* pythonradex (see https://pythonradex.readthedocs.io/en/latest/)

### Installing

Installation is not necessary. Simple download the file [raytracing.py](https://github.com/gica3618/raytracing/blob/main/raytracing.py). Then import it by appending the folder containing raytracing.py to the search path, like this:
```python
import sys
#path to the folder that contains the raytracing.py file:
sys.path.append('/path/to/folder')
import raytracing
```

### Executing the program

Please refer to the example jupyter notebook in the folder [example_notebook](https://github.com/gica3618/raytracing/tree/main/example_notebook).

## Help

If you have any questions or problems, please open an issue here on this github repository.

## Author

Gianni Cataldi

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

If you use this code in your research, please cite this github repository.
