import numpy as np

from openmdao.api import ExplicitComponent

from lsdo_utils.miscellaneous_functions.get_array_indices import get_array_indices


def get_array_expansion_data(shape, expand_indices):
    alphabet = 'abcdefghij'

    in_string = ''
    out_string = ''
    ones_string = ''
    in_shape = []
    out_shape = []
    ones_shape = []
    for index in range(len(shape)):
        if index not in expand_indices:
            in_string += alphabet[index]
            in_shape.append(shape[index])
        else:
            ones_string += alphabet[index]
            ones_shape.append(shape[index])
        out_string += alphabet[index]
        out_shape.append(shape[index])

    einsum_string = '{},{}->{}'.format(in_string, ones_string, out_string)
    in_shape = tuple(in_shape)
    out_shape = tuple(out_shape)
    ones_shape = tuple(ones_shape)

    return einsum_string, in_shape, out_shape, ones_shape


class ArrayExpansionComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('shape', types=tuple)
        self.options.declare('expand_indices', types=list)
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)

    def setup(self):
        shape = self.options['shape']
        expand_indices = self.options['expand_indices']
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        einsum_string, in_shape, out_shape, ones_shape = get_array_expansion_data(shape, expand_indices)

        self.add_input(in_name, shape=in_shape)
        self.add_output(out_name, shape=out_shape)

        in_indices = get_array_indices(*in_shape)
        out_indices = get_array_indices(*out_shape)

        self.einsum_string = einsum_string
        self.ones_shape = ones_shape

        rows = out_indices.flatten()
        cols = np.einsum(einsum_string, in_indices, np.ones(ones_shape, int)).flatten()
        self.declare_partials(out_name, in_name, val=1., rows=rows, cols=cols)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        outputs[out_name] = np.einsum(self.einsum_string, inputs[in_name], np.ones(self.ones_shape))


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    shape = (3,1,4)
    expand_indices = [0,1]

    prob = Problem()
    indeps = IndepVarComp()
    in_name = np.random.rand(4)
    indeps.add_output('in_name', in_name)
    
    prob.model.add_subsystem('indeps', indeps, promotes=['*'])

    prob.model.add_subsystem('array_expansion', ArrayExpansionComp(
        shape=shape, expand_indices=expand_indices, in_name='in_name', out_name='out_name'), promotes=['*']
    )
    
    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)

    print(prob['out_name'])