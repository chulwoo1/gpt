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
grid=g.grid([24, 24, 24, 64], g.double)
#grid=g.grid([48, 48, 48, 96], g.double)
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
    "M5": 1.8,
    "b": 1.5,
    "c": 0.5,
    "Ls": 16,
    "boundary_phases": [1.0, 1.0, 1.0, -1.0],
})

qz = g.qcd.fermion.zmobius( U,
    {
        "mass": 0.00107,
        "M5": 1.8,
        "b": 1.0,
         "c": 0.0,
        "omega": [
            1.0903256131299373  + 1j *( 0 ),
            0.9570283702230611  + 1j *( 0 ),
            0.7048886040934104  + 1j *( 0 ),
            0.48979921782791747  + 1j *( 0 ),
            0.328608311201356  + 1j *( 0 ),
            0.21664245377015995  + 1j *( 0 ),
            0.14121112711957107  + 1j *( 0 ),
            0.0907785101745156  + 1j *( 0 ),
            0.05608303440064219  + 1j *( -0.007537158177840385 ),
            0.05608303440064219  + 1j *( 0.007537158177840385 ),
            0.0365221637144842  + 1j *( -0.03343945161367745 ),
            0.0365221637144842  + 1j *( 0.03343945161367745 ),
        ],
        "boundary_phases": [1.0, 1.0, 1.0, -1.0],
    },
)

fmatrix = qz.converted(g.single)
#fmatrix = exact.converted(g.single)


#op = params["op"](fmatrix)
Mpc = g.qcd.fermion.preconditioner.eo2_ne(parity=g.odd)(fmatrix).Mpc

op = g.algorithms.polynomial.chebyshev({
#    "low"   : 9.644e-7,
    "low"   : 0.000297,
    "high"  : 9.0,
    "order" : 200,
})(Mpc)

grid = op.vector_space[0].grid

# implicitly restarted lanczos
#irl = params["method_evec"]
irl = g.algorithms.eigen.irl({
    "Nk" : 1040,
    "Nstop" : 1000,
    "Nm" : 1200,
    "resid" : 1e-2600,
    "betastp" : 1e-7,
    "maxiter" : 40,
    "Nminres" : 0,
})


# run
start = g.vspincolor(grid)
start[:] = g.vspincolor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
start.checkerboard(g.odd)  # traditionally, calculate odd-site vectors



try:
#    basis, ev = g.load("basis", grids=grid)
    basis, feval = g.load("basis")
except g.LoadError:
#    basis, ev = irl(op, start, params["checkpointer"])
    ckpt1=g.checkpointer("bckpt")
    basis, feval = irl(op, start, ckpt1 )
    g.save("basis", (basis, feval))

g.mem_report()
print("basis done")

dst="."
#grid=g.grid([16, 16, 16, 32], g.double)
#conf = g.default.get("--config", "None")
#U = g.load(conf)
#g.save("config_sav", U, g.format.nersc())

# show available memory
g.mem_report()

# fermion
#q = params["fmatrix"](U)

Mpc = g.qcd.fermion.preconditioner.eo2_ne(parity=g.odd)(fmatrix).Mpc

# load basis vectors
#basis, feval = g.load(params["basis"])
#basis, feval = g.load(f"{dst}/basis")
nbasis = len(basis)

# memory info
g.mem_report()

# norms
for i in range(nbasis):
    g.message("Norm2 of basis[%d] = %g" % (i, g.norm2(basis[i])))

g.mem_report()

# prepare and test basis
for i in range(nbasis):
    g.message(i)
    _, eps2 = g.algorithms.eigen.evals(Mpc, [basis[i]], real=True)
    assert all([e2 < 1e-4 for e2 in eps2])
    g.mem_report(details=False)

# coarse grid
#cgrid = params["cgrid"](q.Mpc.grid[0])
grid=Mpc.vector_space[0].grid
cgrid = g.block.grid(grid,[12,2,2,2,2])
b = g.block.map(cgrid, basis)

# cheby on coarse grid
#cop = params["cmatrix"](q.Mpc, b)
cop = b.coarse_operator(g.algorithms.polynomial.chebyshev({
    "low"   : 0.000684,
    "high"  : 9.0,
    "order" : 200,
})(Mpc))

# implicitly restarted lanczos on coarse grid
#irl = params["method_evec"]
irl = g.algorithms.eigen.irl({
    "Nk" : 2100,
    "Nstop" : 2000,
    "Nm" : 2600,
    "resid" : 1e-10,
    "betastp" : 1e-7,
    "maxiter" : 40,
    "Nminres" : 0,
})

# start vector
cstart = g.vcomplex(cgrid, nbasis)
cstart[:] = g.vcomplex([1] * nbasis, nbasis)

g.mem_report()

# basis
#northo = params["northo"]
northo = 3
for i in range(northo):
    g.message("Orthonormalization round %d" % i)
    b.orthonormalize()

g.mem_report()

# now define coarse-grid operator
g.message(
    "Test precision of promote-project chain: %g"
    % (g.norm2(cstart - b.project * b.promote * cstart) / g.norm2(cstart))
)

g.mem_report()

try:
    cevec, cev = g.load("cevec")
except g.LoadError:
#    cevec, cev = irl(cop, cstart, params["checkpointer"])
    ckpt2=g.checkpointer(f"{dst}/ckpt")
    cevec, cev = irl(cop, cstart,ckpt2)
    g.save("cevec", (cevec, cev))

# smoother
#smoother = params["smoother"](q.Mpc)
smoother = g.algorithms.inverter.cg({
    "eps": 1e-8,
    "maxiter": 15
})(Mpc)

#nsmoother = params["nsmoother"]
nsmoother = 1

v_fine = g.lattice(basis[0])
v_fine_smooth = g.lattice(basis[0])
try:
    ev3 = g.load("ev3")
except g.LoadError:
    ev3 = [0.0] * len(cevec)
    for i, v in enumerate(cevec):
        v_fine @= b.promote * v
        for j in range(nsmoother):
            v_fine_smooth @= smoother * v_fine
            v_fine @= v_fine_smooth / g.norm2(v_fine_smooth) ** 0.5
        ev_smooth, ev_eps2 = g.algorithms.eigen.evals(Mpc, [v_fine], real=True)
        assert ev_eps2[0] < 1e-2
        ev3[i] = ev_smooth[0]
        g.message("Eigenvalue %d = %.15g" % (i, ev3[i]))
    g.save("ev3", ev3)

# save in rbc format
#g.save("lanczos.output", [basis, cevec, ev3], params["format"])
g.save("lanczos.output", [basis, cevec, ev3], g.format.cevec({
    "nsingle" : 10,
    "max_read_blocks" : 16,
}))

# tests
start = g.lattice(basis[0])
start[:] = g.vspincolor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
start *= 1.0 / g.norm2(start) ** 0.5


def save_history(fn, history):
    f = open(fn, "wt")
    for i, v in enumerate(history):
        f.write("%d %.15E\n" % (i, v))
    f.close()


test_solver = params["test_solver"]
solver = g.algorithms.inverter.sequence(
    g.algorithms.inverter.coarse_deflate(cevec, basis, ev3), test_solver
)(Mpc)
v_fine[:] = 0
solver(v_fine, start)
save_history("cg_test.defl_all_ev3", test_solver.history)

solver = g.algorithms.inverter.sequence(
    g.algorithms.inverter.coarse_deflate(cevec[0 : len(basis)], basis, ev3[0 : len(basis)]),
    params["test_solver"],
)(Mpc)
v_fine[:] = 0
solver(v_fine, start)
save_history("cg_test.defl_full", test_solver.history)

v_fine[:] = 0
test_solver(Mpc)(v_fine, start)
save_history("cg_test.undefl", test_solver.history)
