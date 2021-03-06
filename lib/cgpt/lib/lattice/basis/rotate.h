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

template<class VLattice, typename dtype>
void cgpt_basis_rotate(VLattice &basis,dtype* Qt,int j0, int j1, int k0,int k1,int Nm) {
  PMatrix<dtype> _Qt(Qt,Nm);
  basisRotate(basis,_Qt,j0,j1,k0,k1,Nm);
}

template<class Field,class VLattice>
void cgpt_linear_combination(Field &result,VLattice &basis,ComplexD* Qt) {
  typedef typename Field::vector_object vobj;
  GridBase* grid = basis[0].Grid();

  // TODO: map to basisRotateJ
  result.Checkerboard() = basis[0].Checkerboard();

  autoView( result_v , result, AcceleratorWriteDiscard);
  int N = (int)basis.size();

  typedef decltype(basis[0].View(AcceleratorRead)) View;
  Vector<View> basis_v; basis_v.reserve(basis.size());
  for(int k=0;k<basis.size();k++){
    basis_v.push_back(basis[k].View(AcceleratorRead));
  }

#ifndef GRID_HAS_ACCELERATOR
  thread_for(ss, grid->oSites(),{
      vobj B = Zero();
      for(int k=0; k<N; ++k){
	B += Qt[k] * basis_v[k][ss];
      }
      result_v[ss] = B;
    });
#else
  Vector<ComplexD> Qt_jv(N);
  ComplexD * Qt_j = & Qt_jv[0];
  for(int k=0;k<N;++k) Qt_j[k]=Qt[k];
  accelerator_for(ss, grid->oSites(),vobj::Nsimd(),{
      decltype(coalescedRead(basis_v[0][ss])) B;
      B=Zero();
      for(int k=0; k<N; ++k){
	B +=Qt_j[k] * coalescedRead(basis_v[k][ss]);
      }
      coalescedWrite(result_v[ss], B);
    });
#endif

  for(int k=0;k<basis.size();k++) basis_v[k].ViewClose();
}
