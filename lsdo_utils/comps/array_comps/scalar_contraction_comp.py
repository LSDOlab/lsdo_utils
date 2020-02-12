import numpy as np

from openmdao.api import ExplicitComponent

from lsdo_utils.miscellaneous_functions.get_array_indices import get_array_indices


class ScalarContractionComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('shape', types=tuple)
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)

    def setup(self):
        shape = self.options['shape']
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        self.add_input(in_name, shape=shape)
        self.add_output(out_name)

        rows = np.zeros(np.prod(shape), int)
        cols = np.arange(np.prod(shape))
        self.declare_partials(out_name, in_name, val=1., rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        outputs[out_name] = np.sum(inputs[in_name])


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp


    shape = (2, 3)

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('x', np.random.random(shape))
    prob.model.add_subsystem('ivc', comp, promotes=['*'])

    comp = ScalarContractionComp(
        shape=shape,
        in_name='x',
        out_name='y',
    )
    prob.model.add_subsystem('y', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)

    print(prob['x'])
    print(prob['y'])