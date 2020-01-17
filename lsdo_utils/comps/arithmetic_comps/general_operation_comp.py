import numpy as np

from lsdo_utils.comps.array_explicit_component import ArrayExplicitComponent
from lsdo_utils.miscellaneous_functions.process_options import name_types, get_names_list

function_type = type(lambda : None)

class GeneralOperationComp(ArrayExplicitComponent):

    def array_initialize(self):
        self.options.declare('in_names', types=name_types)
        self.options.declare('out_name', types=str)
        self.options.declare('func', types=function_type)
        self.options.declare('deriv', types=function_type)

    def array_setup(self):
        self.options['in_names'] = get_names_list(self.options['in_names'])

        in_names = self.options['in_names']
        out_name = self.options['out_name']
        func = self.options['func']
        deriv = self.options['deriv']

        self.array_add_output(out_name)
        for in_name in in_names:
            self.array_add_input(in_name)
            self.array_declare_partials(out_name, in_name)

    def compute(self, inputs, outputs):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        
        outputs[out_name] = self.options['func'](*[inputs[in_name] for in_name in in_names])

    def compute_partials(self, inputs, partials):
        in_names = self.options['in_names']
        out_name = self.options['out_name']

        result = self.options['deriv'](*[inputs[in_name].flatten() for in_name in in_names])

        for ind, in_name in enumerate(in_names):
            partials[out_name, in_name] = result[ind]


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp


    shape = (2, 3, 4)

    prob = Problem()
    
    comp = IndepVarComp()
    comp.add_output('x', np.random.rand(*shape))
    comp.add_output('y', np.random.rand(*shape))
    comp.add_output('z', np.random.rand(*shape))
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    def func(x, y, z):
        return x * y * z

    def deriv(x, y, z):
        return (
            y * z,
            x * z,
            x * y,
        )

    comp = GeneralOperationComp(
        shape=shape,
        in_names=['x', 'y', 'z'],
        out_name='f',
        func=func,
        deriv=deriv,
    )
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup(check=True)
    prob.run_model()
    prob.check_partials(compact_print=True)

    print(prob['x'] * prob['y'] * prob['z'] - prob['f'])