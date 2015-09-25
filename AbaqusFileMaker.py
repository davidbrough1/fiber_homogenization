from AbaqusGen import generateAbaqusInp
from pymks.datasets import make_microstructure

X = make_microstructure()
generateAbaqusInp('test_microstructure', X[0], elastic_modulus=(120, 80),
                  poissions_ratio=(0.3, 0.3))
