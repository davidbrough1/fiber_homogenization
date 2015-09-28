# import matplotlib.pyplot as plt
# import timeit as tm
import numpy as np

# from pymks.datasets import make_elastic_stress_random
from pymks.datasets import make_microstructure
from AbaqusIO import generateAbaqusInp
from pymks.tools import draw_microstructures
from pymks import MKSHomogenizationModel
from pymks import PrimitiveBasis
from sklearn.cross_validation import train_test_split
# from pymks.tools import draw_goodness_of_fit
from pymks.tools import draw_components
from pymks.stats import autocorrelate
from pymks.tools import draw_autocorrelations
from sklearn.metrics import r2_score
from pymks.tools import draw_goodness_of_fit


sample_size = 200
n_samples = 4*[sample_size]
size = (100, 100)
elastic_modulus = (1.3, 75)
poissons_ratio = (0.42, .22)
macro_strain = 0.001

n_phases = 2
grain_size = [(40, 2), (10, 2), (2, 40), (2, 10)]

grain_size1 = (40, 2, 2)
grain_size2 = (10, 2, 2)
grain_size3 = (2, 40, 2)
grain_size4 = (2, 10, 2)

# Create Samples and calculate stresses

'''dataset, stresses = make_elastic_stress_random(n_samples=n_samples, size=size,
                                               grain_size=grain_size,
                                               elastic_modulus=elastic_modulus,
                                               poissons_ratio=poissons_ratio,
                                               macro_strain=macro_strain,
                                               seed=0)'''

Long_fiber_x = make_microstructure(n_samples=n_samples, size=size,
                                   n_phases=n_phases, grain_size=grain_size1,
                                   seed=1)

Short_fiber_x = make_microstructure(n_samples=n_samples, size=size,
                                    n_phases=n_phases, grain_size=grain_size2,
                                    seed=10)

Long_fiber_y = make_microstructure(n_samples=n_samples, size=size,
                                   n_phases=n_phases, grain_size=grain_size3,
                                   seed=5)

Short_fiber_y = make_microstructure(n_samples=n_samples, size=size,
                                    n_phases=n_phases, grain_size=grain_size4,
                                    seed=3)

dataset = np.concatenate((Long_fiber_x, Long_fiber_y, Short_fiber_x,
                         Short_fiber_y), axis=0)

generateAbaqusInp('Abaqus File', dataset,
                  elastic_modulus=(120, 80),
                  poissions_ratio=(0.3, 0.3))


print dataset.shape
examples = dataset[::sample_size]
# print examples.shape
draw_microstructures((examples))

# Define Model
P_basis = PrimitiveBasis(n_states=2, domain=[0, 1])
model = MKSHomogenizationModel(basis=P_basis,
                               correlations=[(0, 0), (1, 1), (0, 1)])

# Draw 2 point statisitics

'''data_ = P_basis.discretize(dataset)
data_auto = autocorrelate(data_, periodic_axes=(0, 1))
labs = [('Fiber', 'Fiber'), ('Matrix', 'Matrix')]
draw_autocorrelations(data_auto[0], autocorrelations=labs)'''


# Split testing and training segments
flat_shape = (dataset.shape[0],) + (np.prod(dataset.shape[1:]),)

data_train, data_test, stress_train, stress_test = train_test_split(
    dataset.reshape(flat_shape), stresses, test_size=0.2, random_state=3)
# print data_test.shape

# Optimizing polynomial degreee and number of components
'''params_to_tune = {'degree': np.arange(1, 4), 'n_components': np.arange(1, 8)}
fit_params = {'size': dataset[0].shape, 'periodic_axes': [0, 1]}
gs = GridSearchCV(model, params_to_tune, cv=3, n_jobs=3,
                  fit_params=fit_params).fit(data_train, stress_train)

model = gs.best_estimator_'''
model.n_components = 5
model.degree = 3
print('Number of Components'), (model.n_components)
print('Polynomail Order'), (model.degree)

# Fit data to model

model.fit(dataset, stresses, periodic_axes=[0, 1])
shapes = (data_test.shape[0],) + (dataset.shape[1:])

print shapes
data_test = data_test.reshape(shapes)

stress_predict = model.predict(data_test, periodic_axes=[0, 1])
labels = 'Long X', 'Short X', 'Long Y', 'Short Y'

# Draw PCA plot
draw_components([model.reduced_fit_data[:, :2],
                model.reduced_predict_data[:, :2]],
                ['Training Data', 'Testing Data'])


# Draw goodness of fit
fit_data = np.array([stresses, model.predict(dataset, periodic_axes=[0, 1])])
pred_data = np.array([stress_test, stress_predict])
draw_goodness_of_fit(fit_data, pred_data, ['Training Data', 'Testing Data'])

print('R-squared'), (model.score(data_train, stress_train,
                                 periodic_axes=[0, 1]))
