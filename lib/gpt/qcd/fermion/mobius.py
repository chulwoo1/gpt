#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Daniel Richtmann (daniel.richtmann@ur.de)
#                  2022  Tristan Ueding
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
import gpt, copy
from gpt.qcd.fermion.operator import differentiable_fine_operator


class mobius_class_operator(differentiable_fine_operator):
    def __init__(self, name, U, params, otype=None):

        if params["mass"] is not None:
            params["mass_plus"] = params["mass"]
            params["mass_minus"] = params["mass"]

        differentiable_fine_operator.__init__(self, name, U, params, otype)

        def _J5q(dst4d, src5d):
            src4d = gpt.separate(src5d, 0)
            Ls = len(src4d)
            # create correlator at the midpoint of the 5-th direction
            p_plus = gpt.eval(src4d[Ls // 2 - 1] + gpt.gamma[5] * src4d[Ls // 2 - 1])
            p_minus = gpt.eval(src4d[Ls // 2] - gpt.gamma[5] * src4d[Ls // 2])
            gpt.eval(dst4d, 0.5 * (p_plus + p_minus))

        self.J5q = gpt.matrix_operator(
            _J5q,
            vector_space=(self.vector_space_U, self.vector_space_F),
            accept_list=False,
        )

        self.bulk_propagator_to_propagator = self.ExportPhysicalFermionSolution

    def bulk_propagator(self, solver):
        imp = self.ImportPhysicalFermionSource

        inv_matrix = solver(self)

        def prop(dst_sc, src_sc):
            gpt.eval(dst_sc, inv_matrix * imp * src_sc)

        return gpt.matrix_operator(
            prop,
            vector_space=imp.vector_space,
            accept_list=True,
        )


@gpt.params_convention(
    mass=None,
    mass_plus=None,
    mass_minus=None,
    b=None,
    c=None,
    M5=None,
    boundary_phases=None,
    Ls=None,
)
def mobius(U, params):
    params = copy.deepcopy(params)  # save current parameters
    return mobius_class_operator(
        "mobius", U, params, otype=gpt.ot_vector_spin_color(4, 3)
    )
