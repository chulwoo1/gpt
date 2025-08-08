#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2023
#
# New unitary 48I evecs
#
import gpt as g
import os

conf = g.default.get("--config", None)
dst = g.default.get("--dst", None)

U = g.load(conf)

os.makedirs(dst, exist_ok=True)

exact = g.qcd.fermion.mobius(U,{
    "mass": 0.00078,
    "M5": 1.8,
    "b": 1.5,
    "c": 0.5,
    "Ls": 24,
    "boundary_phases": [1.0, 1.0, 1.0, -1.0],
})

sloppy = exact.converted(g.single)

Mpc = g.qcd.fermion.preconditioner.eo2_ne(parity=g.odd)(sloppy).Mpc

grid = Mpc.vector_space[0].grid
cgrid = g.block.grid(grid,[24,4,3,3,4])

op = g.algorithms.polynomial.chebyshev({
    "low"   : 6e-5,
    "high"  : 5.5,
    "order" : 400,
})(Mpc)

irl = g.algorithms.eigen.irl({
    "Nk" : 250,
    "Nstop" : 200,
    "Nm" : 300,
    "resid" : 1e-12,
    "betastp" : 1e-7,
    "maxiter" : 40,
    "Nminres" : 2,
})

start = g.vspincolor(grid)
start[:] = g.vspincolor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
start.checkerboard(g.odd)

if os.path.exists(f"{dst}/basis"):
    basis, ev = g.load(f"{dst}/basis")
    basis = basis[0:200]
    ev = ev[0:200]
else:
    basis, ev = irl(op, start, g.checkpointer(f"{dst}/checkpoint.basis"))
    g.save(f"{dst}/basis", (basis,ev))
    g.barrier()
    # sys.exit(0)

b = g.block.map(cgrid, basis)
for i in range(2):
    g.message("Orthonormalization round %d" % i)
    b.orthonormalize()

nbasis = len(basis)
g.message(f"len(basis) = {nbasis}")

cstart = g.vcomplex(cgrid, nbasis)
cstart[:] = g.vcomplex([1] * nbasis, nbasis)

cop = b.coarse_operator(g.algorithms.polynomial.chebyshev({
    "low"   : 6.0e-4,
    "high"  : 5.5,
    "order" : 200,
})(Mpc))

irl = g.algorithms.eigen.irl({
    "Nk" : 2100,
    "Nstop" : 2000,
    "Nm" : 2200,
    "resid" : 1e-8,
    "betastp" : 1e-7,
    "maxiter" : 40,
    "Nminres" : 1,
    "orthogonalize_nblock" : 64
})

if os.path.exists(f"{dst}/coarse"):
    cevec, cev = g.load(f"{dst}/coarse")
else:
    cevec, cev = irl(cop, cstart, g.checkpointer(f"{dst}/checkpoint.coarse"))
    g.save(f"{dst}/coarse", (cevec, cev))

smoother = g.algorithms.inverter.cg({
    "eps": 1e-8,
    "maxiter": 15
})(Mpc)

nsmoother = 1

v_fine = g.lattice(basis[0])
v_fine_smooth = g.lattice(basis[0])

nblocks = 10
assert len(cevec) % nblocks == 0
evec_per_block = len(cevec) // nblocks
ev3 = [0.0] * len(cevec)

for block in range(nblocks):

    if os.path.exists(f"{dst}/ev3_{block}"):
        ev3 = g.load(f"{dst}/ev3_{block}")
    else:
        for i in range(block*evec_per_block, (block+1)*evec_per_block):
            v = cevec[i]
            g.mem_report(details=False)
            v_fine @= b.promote * v
            for j in range(nsmoother):
                v_fine_smooth @= smoother * v_fine
                v_fine @= v_fine_smooth / g.norm2(v_fine_smooth) ** 0.5
            ev_smooth = g.algorithms.eigen.evals(
                Mpc, [v_fine], calculate_eps2=False, real=True
            )
            ev3[i] = ev_smooth[0]
            g.message("Eigenvalue %d = %.15g" % (i, ev3[i]))
        g.save(f"{dst}/ev3_{block}", ev3)

# now test
start = g.lattice(basis[0])
start[:] = g.vspincolor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
start *= 1.0 / g.norm2(start) ** 0.5

def save_history(fn, history):
    if g.rank() == 0:
        f = open(fn, "wt")
        for i, v in enumerate(history):
            f.write("%d %.15E\n" % (i, v))
        f.close()

test_solver = g.algorithms.inverter.cg({
    "eps": 1e-8,
    "maxiter": 2000
})

solver = g.algorithms.inverter.sequence(
    g.algorithms.inverter.coarse_deflate(cevec, basis, ev3), test_solver
)(Mpc)
v_fine[:] = 0
solver(v_fine, start)
save_history(f"{dst}/cg_test.defl_all_ev3", test_solver.history)

solver = g.algorithms.inverter.sequence(
    g.algorithms.inverter.coarse_deflate(
        cevec[0 : len(basis)], basis, ev3[0 : len(basis)]
    ),
    test_solver,
)(Mpc)
v_fine[:] = 0
solver(v_fine, start)
save_history(f"{dst}/cg_test.defl_full", test_solver.history)

v_fine[:] = 0
test_solver(Mpc)(v_fine, start)
save_history(f"{dst}/cg_test.undefl", test_solver.history)

# save in rbc format
g.save(f"{dst}/lanczos.output", [basis, cevec, ev3], g.format.cevec({
    "nsingle" : 50,
    "max_read_blocks" : 16,
}))
