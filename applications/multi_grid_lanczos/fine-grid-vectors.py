#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Production code to generate fine-grid basis vectorscoarse-grid eigenvectors using existing
#
import gpt as g

# parameters
fn = g.default.get("--params", "params.txt")
#params = g.params(fn, verbose=True)

# load configuration
# U = params["config"]
grid=g.grid([32, 32, 32, 32], g.double)
rng = g.random( "benchmark", "vectorized_ranlux24_24_64" )  
U = g.qcd.gauge.random(grid,rng,scale=0.5 )
conf = g.default.get("--config", "None")
print('conf=',conf)
if conf != "None":
    U = g.load(conf)
g.save("config_sav", U, g.format.nersc())

# matrix to use
#fmatrix = params["fmatrix"](U)
exact = g.qcd.fermion.mobius(U,{
    "mass": 0.00049,
    "M5": 1.4,
    "b": 2.0,
    "c": 1.0,
    "Ls": 12,
    "boundary_phases": [1.0, 1.0, 1.0, -1.0],
})

fmatrix = exact.converted(g.single)

#op = params["op"](fmatrix)
Mpc = g.qcd.fermion.preconditioner.eo2_ne(parity=g.odd)(fmatrix).Mpc

op = g.algorithms.polynomial.chebyshev({
#    "low"   : 9.644e-7,
    "low"   : 0.01,
    "high"  : 5.5,
    "order" : 20,
})(Mpc)

grid = op.vector_space[0].grid

# implicitly restarted lanczos
#irl = params["method_evec"]
irl = g.algorithms.eigen.irl({
    "Nk" : 75,
    "Nstop" : 70,
    "Nm" : 90,
    "resid" : 1e-10,
    "betastp" : 1e-7,
    "maxiter" : 40,
    "Nminres" : 0,
})


# run
start = g.vspincolor(grid)
start[:] = g.vspincolor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
start.checkerboard(g.odd)  # traditionally, calculate odd-site vectors



try:
    basis, ev = g.load("basis", grids=grid)
except g.LoadError:
#    basis, ev = irl(op, start, params["checkpointer"])
    ckpt1=g.checkpointer("bckpt")
    basis, ev = irl(op, start, ckpt1 )
    g.save("basis", (basis, ev))
