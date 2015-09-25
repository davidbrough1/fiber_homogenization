from AbaqusGen import generateAbaqusInp
import numpy as np
from pymks.datasets import make_microstructure

A = np.ndarray(shape = (63,63,63),dtype = 'float')
X = make_microstructure()
generateAbaqusInp('test_microstructure', X[0], elastic_modulus=(120, 80),
                  poissions_ratio=(0.3, 0.3))
