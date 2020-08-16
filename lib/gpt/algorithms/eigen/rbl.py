#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
#          The history of this version of the algorithm is long.  It is mostly based on
#          my earlier version at https://github.com/lehner/Grid/blob/legacy/lib/algorithms/iterative/ImplicitlyRestartedLanczos.h
#          which is based on code from Tom Blum, Taku Izubuchi, Chulwoo Jung and guided by the PhD thesis of Rudy Arthur.
#          I also adopted some of Peter Boyle's convergence test modifications that are in https://github.com/paboyle/Grid .
#
import gpt as g
import numpy as np
import sys

# BLock Lanczos
class irl:
    @g.params_convention(advise=None, mem_report=False)
    def __init__(self, params):
        self.params = params
        self.napply = 0

    def __call__(self, mat, src, ckpt=None):

        # verbosity
        verbose = g.default.is_verbose("irl")

        # checkpointer
        if ckpt is None:
            ckpt = g.checkpointer_none()
        ckpt.grid = src.grid
        self.ckpt = ckpt

        # first approximate largest eigenvalue
        pit = g.algorithms.eigen.power_iteration(eps=0.05, maxiter=10, real=True)
        lambda_max = pit(mat, src)[0]

        # parameters
        Nm = self.params["Nm"]
        Nu = self.params["Nu"]
        Nk = self.params["Nk"]
        Nstop = self.params["Nstop"]
        assert Nm >= Nk and Nstop <= Nk

        # tensors
        dtype = np.float64
        lme = np.zeros((Nu,Nm), dtype)
        lmd = np.zeros((Nu,Nm), dtype)
        lme2 = np.zeros((Nu,Nm), dtype)
#        lmd2 = np.empty((Nu,Nm), dtype)
	Qt = np.zeros((Nm,Nm),dtype)
	Q = np.zeros((Nm,Nm),dtype)
        ev = np.empty((Nm,), dtype)
        ev2 = np.empty((Nm,), dtype)
        q
        ev2_copy = np.empty((Nm,), dtype)

        # fields
        f = g.lattice(src)
        v = g.lattice(src)
	evec = [g.lattice(src) for i in range(Nm)]
        w = [g.lattice(src) for i in range(Nu)]
        w_copy = [g.lattice(src) for i in range(Nu)]

        # advice memory storage
        if not self.params["advise"] is None:
            g.advise(evec, self.params["advise"])

        # scalars
        k1 = 1
        k2 = Nk
        beta_k = 0.0

        # set initial vector
        evec[0] @= src / g.norm2(src) ** 0.5

        # initial Nk steps
        for b in range(Nk/Nu):
            self.blockStep(, lmd, lme, evec, w, w_copy, Nm, b,Nu)

        # restarting loop
        for iter in range(self.params["maxiter"]):
            if verbose:
                g.message("Restart iteration %d" % it)

            Nblock_l = Nblock_k + iter*Nblock_p;
            Nblock_r = Nblock_l + Nblock_p;
            Nl = Nblock_l*Nu;
            Nr = Nblock_r*Nu;
            eval2.resize(Nr);

            for b in range(Nblock_k, Nblock_m):
                self.blockStep(, lmd, lme, evec, w, w_copy, Nm, b,Nu)

            for u in range(Nu):
                for k in range(Nr):
                    lmd2[u][k]=lmd[u][k]
                    lme2[u][k]=lme[u][k]


            Qt = np.identity(Nr, dtype)
            
            # diagonalize
            t0 = g.time()
#            self.diagonalize(ev2, lme2, Nm, Qt)
            self.diagonalize(ev2,lmd2,lme2,Nu,Nr,Qt)
            t1 = g.time()

            if verbose:
                g.message("Diagonalization took %g s" % (t1 - t0))

            # sort
            ev2_copy = ev2.copy()
            ev2 = list(reversed(sorted(ev2)))

            for i in range(Nr):
                g.message("Rval[%d]= %e"%(i,ev2[i]))

            # rotate
            t0 = g.time()
            g.rotate(evec, Qt, k1 - 1, k2 + 1, 0, Nm)
            t1 = g.time()

            if verbose:
                g.message("Basis rotation took %g s" % (t1 - t0))

            # convergence test
            if it >= self.params["Nminres"]:
                if verbose:
                    g.message("Rotation to test convergence")

                # diagonalize
                for k in range(Nm):
                    ev2[k] = ev[k]
                    lme2[k] = lme[k]
                Qt = np.identity(Nm, dtype)

                t0 = g.time()
                self.diagonalize(ev2, lme2, Nk, Qt)
                t1 = g.time()

                if verbose:
                    g.message("Diagonalization took %g s" % (t1 - t0))

                B = g.copy(evec[0])

                allconv = True
                if beta_k >= self.params["betastp"]:
                    jj = 1
                    while jj <= Nstop:
                        j = Nstop - jj
                        g.linear_combination(B, evec[0:Nk], Qt[j, 0:Nk])
                        B *= 1.0 / g.norm2(B) ** 0.5
                        if not ckpt.load(v):
                            mat(v, B)
                            ckpt.save(v)
                        ev_test = g.innerProduct(B, v).real
                        eps2 = g.norm2(v - ev_test * B) / lambda_max ** 2.0
                        if verbose:
                            g.message(
                                "%-65s %-45s %-50s"
                                % (
                                    "ev[ %d ] = %s" % (j, ev2_copy[j]),
                                    "<B|M|B> = %s" % (ev_test),
                                    "|M B - ev B|^2 / ev_max^2 = %s" % (eps2),
                                )
                            )
                        if eps2 > self.params["resid"]:
                            allconv = False
                        if jj == Nstop:
                            break
                        jj = min([Nstop, 2 * jj])

                if allconv:
                    if verbose:
                        g.message("Converged in %d iterations" % it)
                        break

        t0 = g.time()
        g.rotate(evec, Qt, 0, Nstop, 0, Nk)
        t1 = g.time()

        if verbose:
            g.message("Final basis rotation took %g s" % (t1 - t0))

        return (evec[0:Nstop], ev2_copy[0:Nstop])

    def diagonalize(self, eval, lmd, lme, Nu, Nk, Nm, Qt):
        TriDiag = np.zeros((Nk, Nk), dtype=Qt.dtype)
        for u in range(Nu):
            for k in range(Nk):
                TriDiag[k,u+(k/Nu)*Nu] = lmd[u][k];
        for u in range(Nu):
            for k in range(Nk):
                TriDiag[k-Nu,u+(k/Nu)*Nu] = conjugate(lme[u][k-Nu])
                TriDiag[u+(k/Nu)*Nu,k-Nu] = lme[u][k-Nu]
        w, v = np.linalg.eigh(TriDiag)
        for i in range(Nk):
            lmd[Nk - 1 - i] = w[i]
            for j in range(Nk):
                Qt[Nk - 1 - i, j] = v[j, i]

    def blockStep(self, mat, lmd, lme, evec, w, w_copy, Nm, b, Nu):
        assert b*(Nu+1) <= Nm

        verbose = g.default.is_verbose("irl")
        ckpt = self.ckpt

        alph = 0.0
        beta = 0.0

        for k in range (b*Nu,(b+1)*Nu):
             if self.params["mem_report"]:
                g.mem_report(details=False)
# compute
            t0 = g.time()
            mat(w[k-b*Nu], evec[k])
            t1 = g.time()
    
                # allow to restrict maximal number of applications within run
            self.napply += 1
            if "maxapply" in self.params:
                if self.napply == self.params["maxapply"]:
                    if verbose:
                        g.message("Maximal number of matrix applications reached")
                    sys.exit(0)
    
        if b > 0:
            for u in range (Nu):
                for k in range (L-Nu+u,L):
                    w[u] -= coujugage(lme[u][k]) * evec[k]
        for u in range (Nu):
            for k in range (L+u,R):
                lmd[u][k] = g.innerProduct(evec[k],w[u])
                lmd[k-b*Nu][b*Nu+u]=conjugage(lmd[u][k])
            lmd[u][b*Nu+u]=real(lmd[u][b*Nu+u])

        for u in range (Nu):
            for k in range (L+u,R):
                w[u] -= lmd[u][k]*evec[k]
            w_copy[u] = w[u];
        for u in range (Nu):
            for k in range (L,R):
                lme[u][k]=0.;
       
        for u in range (1:Nu):
            g.orthogonalize(w[u],w[0:u])

        for u in range (1:Nu):
            for v in range (u:Nu):
                lme[u][L+v] = g.innerProduct(w[u],w_copy[v])
            lme[u][L+u] = real(lme[u][L+u])
        t3 = g.time()

        for u in range (1:Nu):
            for k in range (L+u,R):
                g.message( 
                    " In block %d, beta[%d][%d]=%e "
                    %( b, u, k-b*Nu,lme[u][k] )
                )
    
#                ckpt.save([w, alph, beta])
    

        if b < Nm/Nu - 1:
            for u in range (1:Nu):
                evec[*Nu+u] @= w[u]
