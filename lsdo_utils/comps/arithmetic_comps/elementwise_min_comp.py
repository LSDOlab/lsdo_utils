import numpy as np
from lsdo_utils.comps.array_explicit_component import ArrayExplicitComponent


class ElementwiseMinComp(ArrayExplicitComponent):

    def array_initialize(self):
        self.options.declare('in_names', types=list)
        self.options.declare('out_name', types=str)
        self.options.declare('rho', types=float)

    def array_setup(self):
        in_names = self.options['in_names']
        out_name = self.options['out_name']

        for in_name in in_names:
            self.array_add_input(in_name)
        self.array_add_output(out_name)
        self.array_declare_partials('*', '*')

    def compute(self, inputs, outputs):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        rho = self.options['rho']

        fmax = -inputs[in_names[0]] - 1
        for in_name in in_names:
            fmax = np.maximum(fmax, -inputs[in_name])

        arg = 0.
        for in_name in in_names:
            arg += np.exp(rho * (-inputs[in_name] - fmax))

        outputs[out_name] = -(
            fmax + 1. / rho * np.log(arg)
        )

    def compute_partials(self, inputs, partials):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        rho = self.options['rho']

        fmax = -inputs[in_names[0]] - 1
        for in_name in in_names:
            fmax = np.maximum(fmax, -inputs[in_name])

        arg = 0.
        for in_name in in_names:
            arg += np.exp(rho * (-inputs[in_name] - fmax))

        for in_name in in_names:
            partials[out_name, in_name] = (
                    1. / arg * np.exp(rho * (-inputs[in_name] - fmax))
            ).flatten()


if __name__ == '__main__':
    from openmdao.api import Problem, Group, IndepVarComp

    in1 = 12.3
    in2 = 12.
    in3 = 12.1
    rho = 20.
    shape = (2, 3)

    prob = Problem()

    model = Group()

    comp = IndepVarComp()
    comp.add_output('in1', in1, shape=shape)
    comp.add_output('in2', in2, shape=shape)
    comp.add_output('in3', in3, shape=shape)
    model.add_subsystem('ivc', comp, promotes=['*'])

    comp = ElementwiseMinComp(shape=shape, in_names=['in1', 'in2', 'in3'], out_name='out', rho=rho)
    model.add_subsystem('comp', comp, promotes=['*'])

    prob.model = model
    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)
    for var_name in ['in1', 'in2', 'in3', 'out']:
        print(var_name, prob[var_name])