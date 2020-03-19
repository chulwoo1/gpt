/*
  CGPT

  Authors: Christoph Lehner 2020
*/
#include "lib.h"

EXPORT(create_grid,{
    
    PyObject* _gdimension, * _precision, * _type;
    if (!PyArg_ParseTuple(args, "OOO", &_gdimension, &_precision,&_type)) {
      return NULL;
    }
    
    std::vector<int> gdimension;
    std::string precision, type;
    
    cgpt_convert(_gdimension,gdimension);
    cgpt_convert(_precision,precision);
    cgpt_convert(_type,type);
    
    int nd = (int)gdimension.size();
    int Nsimd;
    
    if (precision == "single") {
      Nsimd = vComplexF::Nsimd();
    } else if (precision == "double") {
      Nsimd = vComplexD::Nsimd();
    } else {
      ERR("Unknown precision");
    }
    
    GridBase* grid;
    if (nd >= 4) {
      std::vector<int> gdimension4d = gdimension; gdimension4d.resize(4);
      GridCartesian* grid4d = SpaceTimeGrid::makeFourDimGrid(gdimension4d, GridDefaultSimd(4,Nsimd), GridDefaultMpi());
      if (nd == 4) {
	if (type == "redblack") {
	  grid = SpaceTimeGrid::makeFourDimRedBlackGrid(grid4d);
	  delete grid4d;
	} else if (type == "full") {
	  grid = grid4d;
	} else {
	  ERR("Unknown grid type");
	}
      } else if (nd == 5) {
	if (type == "redblack") {
	  grid = SpaceTimeGrid::makeFiveDimRedBlackGrid(gdimension[5],grid4d);
	  delete grid4d;
	} else if (type == "full") {
	  grid = SpaceTimeGrid::makeFiveDimGrid(gdimension[5],grid4d);
	  delete grid4d;
	} else {
	  ERR("Unknown grid type");
	}	
      } else if (nd > 5) {
	ERR("Unknown dimension");
      }

    } else {
      // TODO: give gpt full control over mpi,simd,cbmask?
      // OR: at least give user option to make certain dimensions not simd/mpi directions
      std::cerr << "Unknown dimension " << nd << std::endl;
      assert(0);
    }
    
    return PyLong_FromVoidPtr(grid);
    
  });



EXPORT(delete_grid,{
    
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    ((GridCartesian*)p)->Barrier(); // before a grid goes out of life, we need to synchronize
    delete ((GridCartesian*)p);
    return PyLong_FromLong(0);
    
  });



EXPORT(grid_barrier,{
    
    void* p;
    if (!PyArg_ParseTuple(args, "l", &p)) {
      return NULL;
    }
    
    ((GridCartesian*)p)->Barrier();
    return PyLong_FromLong(0);
    
  });
  


EXPORT(grid_globalsum,{
    
    void* p;
    PyObject* o;
    if (!PyArg_ParseTuple(args, "lO", &p,&o)) {
      return NULL;
    }
    
    GridCartesian* grid = (GridCartesian*)p;
    if (PyComplex_Check(o)) {
      ComplexD c;
      cgpt_convert(o,c);
      grid->GlobalSum(c);
      return PyComplex_FromDoubles(c.real(),c.imag());
    } else if (PyFloat_Check(o)) {
      RealD c;
      cgpt_convert(o,c);
      grid->GlobalSum(c);
      return PyFloat_FromDouble(c);
    } else if (PyLong_Check(o)) {
      uint64_t c;
      cgpt_convert(o,c);
      grid->GlobalSum(c);
      return PyLong_FromLong(c);
    } else if (PyArray_Check(o)) {
      PyArrayObject* ao = (PyArrayObject*)o;
      int dt = PyArray_TYPE(ao);
      void* data = PyArray_DATA(ao);
      size_t nbytes = PyArray_NBYTES(ao);
      if (dt == NPY_FLOAT32 || dt == NPY_COMPLEX64) {
	grid->GlobalSumVector((RealF*)data, nbytes / 4);
      } else if (dt == NPY_FLOAT64 || NPY_COMPLEX128) {
	grid->GlobalSumVector((RealD*)data, nbytes / 8);
      } else {
	ERR("Unsupported numy data type (single, double, csingle, cdouble currently allowed)");
      }
    } else {
      ERR("Unsupported object");
    }
    // need to act on floats, complex, and numpy arrays PyArrayObject
    //PyArrayObject* p;
    return PyLong_FromLong(0);
  });
