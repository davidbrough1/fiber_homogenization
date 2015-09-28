

from pymks.datasets import make_microstructure
from AbaqusIO import generateAbaqusInp

n_phases = 2
grain_size1 = (40, 2, 2)

size = (100, 100,100)
Long_fiber_x = make_microstructure(n_samples=1, size=size,
                                   n_phases=n_phases, grain_size=grain_size1,
                                   seed=1)
Ab_fiber = Long_fiber_x+1
generateAbaqusInp('test_microstructure', Ab_fiber[0],
                  elastic_modulus=(120, 80),
                  poissions_ratio=(0.3, 0.3))
print "Done"
