import numpy as np

from openmdao.api import ExplicitComponent


class ArrayReorderComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('in_shape', types=tuple)
        self.options.declare('out_shape', types=tuple)
        self.options.declare('in_subscripts', types=str)
        self.options.declare('out_subscripts', types=str)
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)

    def setup(self):
        in_shape = self.options['in_shape']
        out_shape = self.options['out_shape']
        in_subscripts = self.options['in_subscripts']
        out_subscripts = self.options['out_subscripts']
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        self.add_input(in_name, shape=in_shape)
        self.add_output(out_name, shape=out_shape)

        size = np.prod(in_shape)
        rows = np.arange(size)
        cols = np.einsum(
            in_subscripts + '->' + out_subscripts, 
            np.arange(size).reshape(in_shape),
        ).flatten()
        self.declare_partials(out_name, in_name, val=1., rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        in_subscripts = self.options['in_subscripts']
        out_subscripts = self.options['out_subscripts']
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        
        outputs[out_name] = np.einsum(in_subscripts + '->' + out_subscripts, inputs[in_name])