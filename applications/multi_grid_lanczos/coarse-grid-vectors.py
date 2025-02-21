#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020
#
# Production code to generate coarse-grid eigenvectors using existing
# fine-grid basis vectors
#
import gpt as g

# show available memory
g.mem_report()

# parameters
#fn = g.default.get("--params", "params.txt")
#params = g.params(fn, verbose=True)

# load configuration
#U = params["config"]
grid=g.grid([16, 16, 16, 32], g.double)
conf = g.default.get("--config", "None")
U = g.load(conf)
g.save("config_sav", U, g.format.nersc())

# show available memory
g.mem_report()

# fermion
#q = params["fmatrix"](U)
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
Mpc = g.qcd.fermion.preconditioner.eo2_ne(parity=g.odd)(fmatrix).Mpc


# load basis vectors
#nbasis = params["nbasis"]
nbasis=20
if 1:
    fg_basis, fg_cevec, fg_feval = g.load(
        nbasis,
        grids=grid,
        nmax=nbasis,
        advise_basis=g.infrequent_use,
        advise_cevec=g.infrequent_use,
    )
else:
    fg_basis, fg_feval = g.load(f"{dst}/basis")

# memory info
g.mem_report()

# norms
for i in range(nbasis):
    g.message("Norm2 of basis[%d] = %g" % (i, g.norm2(fg_basis[i])))

for i in range(nbasis):
    g.message("Norm2 of cevec[%d] = %g" % (i, g.norm2(fg_cevec[i])))

g.mem_report()

# prepare and test basis
basis = []
assert nbasis > 0
b = g.block.map(fg_cevec[0].grid, fg_basis)
for i in range(nbasis):
    basis.append(
        g.vspincolor(fg_basis[0].grid)
    )  # don't advise yet, let it be first touched on accelerator
    g.message(i)
    if i < params["nbasis_on_host"]:
        g.message("marked as infrequent use")
        # basis[i].advise( g.infrequent_use )

    basis[i] @= b.promote * fg_cevec[i]
    _, ev_eps2 = g.algorithms.eigen.evals(Mpc, [basis[i]], real=True)
    assert ev_eps2[0] < 1e-4
    g.message("Compare to: %g" % fg_feval[i])

    g.mem_report(details=False)

# now discard original basis
del fg_basis
del fg_cevec
g.message("Memory information after discarding original basis:")
g.mem_report()

# coarse grid
#cgrid = params["cgrid"](basis[0].grid)
cgrid = g.block.grid(grid,[12,4,4,4,4])

b = g.block.map(cgrid, basis)

# cheby on coarse grid
cop = params["cmatrix"](q.Mpc, b)

# implicitly restarted lanczos on coarse grid
#irl = params["method_evec"]
irl = g.algorithms.eigen.irl({
    "Nk" : 100,
    "Nstop" : 120,
    "Nm" : 200,
    "resid" : 1e-8,
    "betastp" : 1e-8,
    "maxiter" : 40,
    "Nminres" : 0,
})

# start vector
cstart = g.vcomplex(cgrid, nbasis)
cstart[:] = g.vcomplex([1] * nbasis, nbasis)

g.mem_report()

# basis
northo = params["northo"]
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
    cevec, cev = irl(cop, cstart, params["checkpointer"])
    g.save("cevec", (cevec, cev))

# smoother
smoother = params["smoother"](q.Mpc)
nsmoother = params["nsmoother"]
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
        ev_smooth, ev_eps2 = g.algorithms.eigen.evals(q.Mpc, [v_fine], real=True)
        assert ev_eps2[0] < 1e-2
        ev3[i] = ev_smooth[0]
        g.message("Eigenvalue %d = %.15g" % (i, ev3[i]))
    g.save("ev3", ev3)

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
)(q.Mpc)
v_fine[:] = 0
solver(v_fine, start)
save_history("cg_test.defl_all_ev3", test_solver.history)

solver = g.algorithms.inverter.sequence(
    g.algorithms.inverter.coarse_deflate(cevec[0 : len(basis)], basis, ev3[0 : len(basis)]),
    params["test_solver"],
)(q.Mpc)
v_fine[:] = 0
solver(v_fine, start)
save_history("cg_test.defl_full", test_solver.history)

v_fine[:] = 0
test_solver(q.Mpc)(v_fine, start)
save_history("cg_test.undefl", test_solver.history)

# save in rbc format
g.save("lanczos.output", [basis, cevec, ev3], params["format"])
