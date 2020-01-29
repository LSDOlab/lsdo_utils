import numpy as np

from lsdo_utils.comps.array_explicit_component import ArrayExplicitComponent
from lsdo_utils.miscellaneous_functions.process_options import name_types, get_names_list
from lsdo_utils.miscellaneous_functions.process_options import scalar_types, get_scalars_list


class LinearCombinationComp(ArrayExplicitComponent):

    def array_initialize(self):
        self.options.declare('out_name', types=str)
        self.options.declare('in_names', default=None, types=name_types, allow_none=True)
        self.options.declare('coeffs', default=1., types=scalar_types)
        self.options.declare('coeffs_dict', default=None, types=dict, allow_none=True)
        self.options.declare('constant', default=0., types=(int, float, np.ndarray))

        self.post_initialize()

    def post_initialize(self):
        pass

    def pre_setup(self):
        pass

    def array_setup(self):
        self.pre_setup()

        if self.options['coeffs_dict']:
            self.options['in_names'] = []
            self.options['coeffs'] = []
            for in_name in self.options['coeffs_dict']:
                coeff = self.options['coeffs_dict'][in_name]
                self.options['in_names'].append(in_name)
                self.options['coeffs'].append(coeff)
        else:
            self.options['in_names'] = get_names_list(self.options['in_names'])
            self.options['coeffs'] = get_scalars_list(self.options['coeffs'], self.options['in_names'])

        in_names = self.options['in_names']
        out_name = self.options['out_name']
        coeffs = self.options['coeffs']
        constant = self.options['constant']

        self.array_add_output(out_name)
        for in_name, coeff in zip(in_names, coeffs):
            self.array_add_input(in_name)
            self.array_declare_partials(out_name, in_name, val=coeff)

    def compute(self, inputs, outputs):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        coeffs = self.options['coeffs']
        constant = self.options['constant']
        
        outputs[out_name] = constant
        for in_name, coeff in zip(in_names, coeffs):
            outputs[out_name] += coeff * inputs[in_name]


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp


    shape = (2, 3, 4)

    prob = Problem()
    
    comp = IndepVarComp()
    comp.add_output('x', np.random.rand(*shape))
    comp.add_output('y', np.random.rand(*shape))
    comp.add_output('z', np.random.rand(*shape))
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = LinearCombinationComp(
        shape=shape,
        in_names=['x', 'y', 'z'],
        out_name='f',
        coeffs=[1., -2., 3.],
        constant=1.5,
    )
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup(check=True)
    prob.run_model()
    prob.check_partials(compact_print=True)

    print(1.5 + 1 * prob['x'] - 2 * prob['y'] + 3 * prob['z'] - prob['f'])