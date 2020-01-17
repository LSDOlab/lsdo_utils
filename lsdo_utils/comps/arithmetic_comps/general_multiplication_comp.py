import numpy as np

from lsdo_utils.comps.array_explicit_component import ArrayExplicitComponent
from lsdo_utils.miscellaneous_functions.process_options import name_types, get_names_list
from lsdo_utils.miscellaneous_functions.process_options import scalar_types, get_scalars_list


class GeneralMultiplicationComp(ArrayExplicitComponent):

    def array_initialize(self):
        self.options.declare('in_names', types=name_types)
        self.options.declare('out_name', types=str)
        self.options.declare('powers', default=1., types=scalar_types)
        self.options.declare('constant', default=1., types=(int, float, np.ndarray))

    def array_setup(self):
        self.options['in_names'] = get_names_list(self.options['in_names'])
        self.options['powers'] = get_scalars_list(self.options['powers'], self.options['in_names'])

        in_names = self.options['in_names']
        out_name = self.options['out_name']
        powers = self.options['powers']
        constant = self.options['constant']

        self.array_add_output(out_name)
        for in_name in in_names:
            self.array_add_input(in_name)
            self.array_declare_partials(out_name, in_name)

    def compute(self, inputs, outputs):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        powers = self.options['powers']
        constant = self.options['constant']
        
        outputs[out_name] = constant
        for in_name, power in zip(in_names, powers):
            outputs[out_name] *= inputs[in_name] ** power

    def compute_partials(self, inputs, partials):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        powers = self.options['powers']
        constant = self.options['constant']
        
        value = constant
        for in_name, power in zip(in_names, powers):
            value *= inputs[in_name] ** power

        for in_name in in_names:
            deriv = constant * np.ones(self.options['shape'])
            for in_name2, power in zip(in_names, powers):
                a = 1.
                b = power
                if in_name == in_name2:
                    a = power
                    b = power - 1.
                deriv *= a * inputs[in_name2] ** b

            partials[out_name, in_name] = deriv.flatten()


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp


    shape = (2, 3, 4)

    prob = Problem()
    
    comp = IndepVarComp()
    comp.add_output('x', np.random.rand(*shape))
    comp.add_output('y', np.random.rand(*shape))
    comp.add_output('z', np.random.rand(*shape))
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = GeneralMultiplicationComp(
        shape=shape,
        in_names=['x', 'y', 'z'],
        out_name='f',
        powers=[1., -2., 3.],
        constant=1.5,
    )
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup(check=True)
    prob.run_model()
    prob.check_partials(compact_print=True)

    print(1.5 * prob['x'] ** 1 * prob['y'] ** -2 * prob['z'] ** 3 - prob['f'])