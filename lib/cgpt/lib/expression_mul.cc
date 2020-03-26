/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/
#include "lib.h"

#include "expression/matmul.h"
#include "expression/mul.h"

#define PER_TENSOR_TYPE(T)						\
  INSTANTIATE(T,vComplexF)						\
  INSTANTIATE(T,vComplexD)						\

#define INSTANTIATE(T,vtype)						\
  template cgpt_Lattice_base* cgpt_lattice_mul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice<T<vtype>>& la,int unary_b, cgpt_Lattice_base* b, int unary_expr); \
  template cgpt_Lattice_base* cgpt_lattice_matmul(cgpt_Lattice_base* dst, bool ac, int unary_a, Lattice<T<vtype>>& la, PyArrayObject* b, std::string& bot, int unary_b, int unary_expr, bool reverse);

#include "tensors.h"

#undef PER_TENSOR_TYPE
