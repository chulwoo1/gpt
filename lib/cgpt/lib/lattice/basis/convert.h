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

template<typename T>
void cgpt_basis_fill(PVector<Lattice<T>>& basis, std::vector<cgpt_Lattice_base*>& _basis) {
  basis.resize(_basis.size());
  for (size_t i=0;i<basis.size();i++)
    basis(i) = &compatible<T>(_basis[i])->l;
}

static void cgpt_basis_fill(std::vector<cgpt_Lattice_base*>& basis, PyObject* _basis, int idx) {
  ASSERT(PyList_Check(_basis));
  Py_ssize_t size = PyList_Size(_basis);
  basis.resize(size);
  for (Py_ssize_t i=0;i<size;i++) {
    PyObject* li = PyList_GetItem(_basis,i);
    PyObject* v_obj = PyObject_GetAttrString(li,"v_obj");
    ASSERT(v_obj && PyList_Check(v_obj));
    ASSERT(idx >= 0 && idx < PyList_Size(v_obj));
    PyObject* obj = PyList_GetItem(v_obj,idx);
    ASSERT(PyLong_Check(obj));
    basis[i] = (cgpt_Lattice_base*)PyLong_AsVoidPtr(obj);
  }
}

