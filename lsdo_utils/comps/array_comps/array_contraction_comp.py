import numpy as np

from openmdao.api import ExplicitComponent

from lsdo_utils.miscellaneous_functions.get_array_indices import get_array_indices
from lsdo_utils.miscellaneous_functions.decompose_shape_tuple import decompose_shape_tuple


class ArrayContractionComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('shape', types=tuple)
        self.options.declare('contract_indices', types=list)
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)

    def setup(self):
        shape = self.options['shape']
        contract_indices = self.options['contract_indices']
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        (
            out_string, ones_string, in_string,
            out_shape, ones_shape, in_shape,
        ) = decompose_shape_tuple(shape, contract_indices)

        einsum_string = '{},{}->{}'.format(out_string, ones_string, in_string)

        self.add_input(in_name, shape=in_shape)
        self.add_output(out_name, shape=out_shape)

        in_indices = get_array_indices(*in_shape)
        out_indices = get_array_indices(*out_shape)

        rows = np.einsum(einsum_string, out_indices, np.ones(ones_shape, int)).flatten()
        cols = in_indices.flatten()
        self.declare_partials(out_name, in_name, val=1., rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        contract_indices = self.options['contract_indices']

        outputs[out_name] = np.sum(inputs[in_name], axis=tuple(contract_indices))


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    shape = (3, 2, 4)
    contract_indices = [0, 1]

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('in_name', np.random.random(shape))
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = ArrayContractionComp(
        shape=shape, 
        contract_indices=contract_indices,
        out_name='out_name',
        in_name='in_name', 
    )
    prob.model.add_subsystem('array_contraction_comp', comp, promotes=['*'])
    
    prob.setup(check=True)
    prob.run_model()
    prob.check_partials(compact_print=True)

    print(prob['in_name'])
    print(prob['out_name'])