from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from lsdo_utils.miscellaneous_functions.get_array_indices import get_array_indices


alphabet = 'abcdefghij'

def insert_3_into_tuple(shape, index):
    return shape[:index] + (3,) + shape[index:]

class CrossProductComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('shape_no_3', types=tuple)
        self.options.declare('in1_index', types=int)
        self.options.declare('in2_index', types=int)
        self.options.declare('out_index', types=int)
        self.options.declare('in1_name', types=str)
        self.options.declare('in2_name', types=str)
        self.options.declare('out_name', types=str)

    def setup(self):
        shape_no_3 = self.options['shape_no_3']
        in1_index = self.options['in1_index']
        in2_index = self.options['in2_index']
        out_index = self.options['out_index']
        in1_name = self.options['in1_name']
        in2_name = self.options['in2_name']
        out_name = self.options['out_name']

        in1_shape = insert_3_into_tuple(shape_no_3, in1_index)
        in2_shape = insert_3_into_tuple(shape_no_3, in2_index)
        out_shape = insert_3_into_tuple(shape_no_3, out_index)

        self.add_input(in1_name, shape=in1_shape)
        self.add_input(in2_name, shape=in2_shape)
        self.add_output(out_name, shape=out_shape)

        in1_indices = get_array_indices(*in1_shape)
        in2_indices = get_array_indices(*in2_shape)
        out_indices = get_array_indices(*out_shape)

        ones = np.ones(3, int)

        rank = len(shape_no_3)
        def get_einsum_string_rows(index):
            return '{}m{},n->{}{}mn'.format(
                alphabet[:index], alphabet[index:rank],
                alphabet[:index], alphabet[index:rank],
            )
        def get_einsum_string_cols(index):
            return '{}m{},n->{}{}nm'.format(
                alphabet[:index], alphabet[index:rank],
                alphabet[:index], alphabet[index:rank],
            )

        rows = np.einsum(
            get_einsum_string_rows(out_index),
            out_indices, ones,
        ).flatten()
        cols = np.einsum(
            get_einsum_string_cols(in1_index),
            in1_indices, ones,
        ).flatten()
        self.declare_partials(out_name, in1_name, rows=rows, cols=cols)

        rows = np.einsum(
            get_einsum_string_rows(out_index),
            out_indices, ones,
        ).flatten()
        cols = np.einsum(
            get_einsum_string_cols(in2_index),
            in2_indices, ones,
        ).flatten()
        self.declare_partials(out_name, in2_name, rows=rows, cols=cols)

        # rows = np.einsum('i...,j->i...j', indices, ones).flatten()
        # cols = np.einsum('j...,i->i...j', indices, ones).flatten()
        # self.declare_partials(out_name, in1_name, rows=rows, cols=cols)
        # self.declare_partials(out_name, in2_name, rows=rows, cols=cols)

        # self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        in1_index = self.options['in1_index']
        in2_index = self.options['in2_index']
        out_index = self.options['out_index']
        in1_name = self.options['in1_name']
        in2_name = self.options['in2_name']
        out_name = self.options['out_name']

        outputs[out_name] = np.cross(
            inputs[in1_name], inputs[in2_name], 
            axisa=in1_index,
            axisb=in2_index,
            axisc=out_index,
        )

    def compute_partials(self, inputs, partials):
        shape_no_3 = self.options['shape_no_3']
        in1_index = self.options['in1_index']
        in2_index = self.options['in2_index']
        out_index = self.options['out_index']
        in1_name = self.options['in1_name']
        in2_name = self.options['in2_name']
        out_name = self.options['out_name']

        ones = np.ones(3)
        eye = np.eye(3)
        rank = len(shape_no_3)

        tmps = {0: None, 1: None, 2: None}
        for ind in range(3):
            array = np.einsum(
                '...,m->...m',
                np.ones(shape_no_3), 
                eye[ind, :],
            )
            
            array = np.einsum(
                '...,m->...m',
                np.cross(
                    np.einsum(
                        '...,m->...m',
                        np.ones(shape_no_3), 
                        eye[ind, :],
                    ), 
                    inputs[in2_name], 
                    axisa=-1,
                    axisb=in2_index,
                    axisc=-1,
                ),
                eye[ind, :],
            )

            tmps[ind] = array

        partials[out_name, in1_name] = (tmps[0] + tmps[1] + tmps[2]).flatten()

        tmps = {0: None, 1: None, 2: None}
        for ind in range(3):
            array = np.einsum(
                '...,m->...m',
                np.ones(shape_no_3), 
                eye[ind, :],
            )
            
            array = np.einsum(
                '...,m->...m',
                np.cross(
                    inputs[in1_name], 
                    np.einsum(
                        '...,m->...m',
                        np.ones(shape_no_3), 
                        eye[ind, :],
                    ), 
                    axisa=in1_index,
                    axisb=-1,
                    axisc=-1,
                ),
                eye[ind, :],
            )

            tmps[ind] = array

        partials[out_name, in2_name] = (tmps[0] + tmps[1] + tmps[2]).flatten()


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp


    shape = (2, 4, 5)
    in1_index = 1
    in2_index = 2
    out_index = 3

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('in1', val=np.random.random(insert_3_into_tuple(shape, in1_index)))
    comp.add_output('in2', val=np.random.random(insert_3_into_tuple(shape, in2_index)))
    prob.model.add_subsystem('ivc', comp, promotes=['*'])

    comp = CrossProductComp(
        shape_no_3=shape,  
        out_index=out_index,
        in1_index=in1_index,
        in2_index=in2_index,
        out_name='out',
        in1_name='in1', 
        in2_name='in2',
    )
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup(check=True)
    prob.run_model()
    prob.check_partials(compact_print=True)