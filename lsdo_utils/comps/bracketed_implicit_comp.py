import numpy as np

from openmdao.api import ImplicitComponent, NewtonSolver, DirectSolver

from lsdo_utils.miscellaneous.simple_types import function_type


class BracketedImplicitComp(ImplicitComponent):

    def initialize(self):
        self.options.declare('shape', types=tuple)
        self.options.declare('out_name', types=str)
        self.options.declare('in_names', types=list)
        self.options.declare('get_res_func', types=function_type)
        self.options.declare('get_derivs_func', types=function_type)
        self.options.declare('max_iter', default=50, types=int)

        self.post_initialize()

    def post_initialize(self):
        pass

    def pre_setup(self):
        pass

    def setup(self):
        self.pre_setup()

        shape = self.options['shape']
        out_name = self.options['out_name']
        in_names = self.options['in_names']

        for in_name in in_names:
            self.add_input(in_name, shape=shape)
        self.add_output(out_name, shape=shape)

        arange = np.arange(np.prod(shape))

        self.declare_partials('*', '*', rows=arange, cols=arange)

    def apply_nonlinear(self, inputs, outputs, residuals):
        out_name = self.options['out_name']
        get_res_func = self.options['get_res_func']

        residuals[out_name] = get_res_func(self.options, inputs, outputs[out_name])

    def solve_nonlinear(self, inputs, outputs):
        out_name = self.options['out_name']
        get_res_func = self.options['get_res_func']

        xp = np.zeros(self.options['shape'])
        xn = np.ones(self.options['shape'])

        for ind in range(self.options['max_iter']):
            x = 0.5 * xp + 0.5 * xn
            r = get_res_func(self.options, inputs, x)
            mask_p = r >= 0
            mask_n = r < 0
            xp[mask_p] = x[mask_p]
            xn[mask_n] = x[mask_n]

        outputs[out_name] = 0.5 * xp + 0.5 * xn

    def linearize(self, inputs, outputs, partials):
        out_name = self.options['out_name']
        get_derivs_func = self.options['get_derivs_func']

        self.jac = get_derivs_func(self.options, inputs, outputs[out_name], partials)

    def solve_linear(self, d_outputs, d_residuals, mode):
        out_name = self.options['out_name']

        if mode == 'fwd':
            d_outputs[out_name] += 1. / self.jac * d_residuals[out_name]
        else:
            d_residuals[out_name] += 1. / self.jac * d_outputs[out_name]