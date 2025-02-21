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
grid=g.grid([16, 16, 16, 32], g.double)
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
    "mass": 0.01,
    "M5": 1.4,
    "b": 2.0,
    "c": 1.0,
    "Ls": 12,
    "boundary_phases": [1.0, 1.0, 1.0, -1.0],
})

qz = g.qcd.fermion.zmobius( U,
    {
        "mass": 0.01,
        "M5": 1.8,
        "b": 1.0,
        "c": 0.0,
        "omega": [
            1.45806438985048 + 1j *(0),
            1.18231318389348 + 1j *(0),
            0.830951166685955 + 1j *(0),
            0.542352409156791 + 1j *(0),
            0.341985020453729 + 1j *(0),
            0.21137902619029 + 1j *(0),
            0.126074299502912 + 1j *(0),
            0.0990136651962626 + 1j *(0),
            0.0686324988446592 + 1j *(0.0550658530827402),
            0.0686324988446592 + 1j *(0.0550658530827402),
        ],
        "boundary_phases": [1.0, 1.0, 1.0, -1.0],
    },
)

fmatrix = qz.converted(g.single)

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
    "Nk" : 25,
    "Nstop" : 20,
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
