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

# Restarted Block  Lanczos
class rbl:
    @g.params_convention(
        orthogonalize_nblock=4,
        mem_report=False,
        rotate_use_accelerator=True,
        Nm=None,
        Nk=None,
        Nu=None,
        Nstop=None,
        resid=None,
        betastp=None,
        maxiter=None,
        Nminres=None,
    )
    def __init__(self, params):
        self.params = params
        self.napply = 0

    def __call__(self, mat, src, ckpt=None):

        # verbosity
        verbose = g.default.is_verbose("rbl")

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
        Np = Nm-Nk
        MaxIter=self.params["maxiter"]
        Np /= MaxIter
        print ( 'Nm=',Nm,'Nu=',Nu,'Nk=',Nk )
        assert Nm >= Nk and Nstop <= Nk
        assert Np >= Nu

        # tensors
        dtype = np.float64
        ctype = np.complex128
         
        lme = np.zeros((Nu,Nm), ctype)
        lmd = np.zeros((Nu,Nm), ctype)
        lme2 = np.zeros((Nu,Nm), ctype)
        lmd2 = np.empty((Nu,Nm), ctype)
        Qt = np.zeros((Nm,Nm),ctype)
        Q = np.zeros((Nm,Nm),ctype)
        ev = np.empty((Nm,), dtype)
        ev2_copy = np.empty((Nm,), dtype)

        # fields
        f = g.lattice(src)
        v = g.lattice(src)
        evec = [g.lattice(src) for i in range(Nm)]
        w = [g.lattice(src) for i in range(Nu)]
        w_copy = [g.lattice(src) for i in range(Nu)]

        # advice memory storage
#        if not self.params["advise"] is None:
#            g.advise(evec, self.params["advise"])

        # scalars
        k1 = 1
        k2 = Nk
        beta_k = 0.0

        rng=g.random("test")
        # set initial vector
#        rng.zn(w)
        for i in range(Nu):
            rng.zn(w[i])
            if i > 0: 
                g.orthogonalize(w[i],evec[0:i])
            evec[i]=g.copy(w[i])
            evec[i] *= 1.0/ g.norm2(evec[i]) ** 0.5
            if verbose:
                g.message("norm(evec[%d]=%e "%(i,g.norm2(evec[i])))
            if i > 0: 
                for j in range(i):
                    ip=g.inner_product(evec[j],w[i])
                    if np.abs(ip) >1e-6:
                        g.message("inner(evec[%d],w[%d])=%e %e"% (j,i,ip.real,ip.imag))
#           evec[i] @= src[i] / g.norm2(src[i]) ** 0.5

        # initial Nk steps
        Nblock_k = int(Nk/Nu)
        for b in range(Nblock_k):
            self.blockStep(mat, lmd, lme, evec, w, w_copy, Nm, b,Nu)

        Nblock_p = int(Np/Nu)
        # restarting loop
#        for it in range(self.params["maxiter"]):
        for it in range(MaxIter):
#            if verbose:
            g.message("Restart iteration %d" % it)

            Nblock_l = Nblock_k + it*Nblock_p;
            Nblock_r = Nblock_l + Nblock_p;
            Nl = Nblock_l*Nu
            Nr = Nblock_r*Nu
#           ev2.resize(Nr)
            ev2 = np.empty((Nr,), dtype)

            for b in range(Nblock_l, Nblock_r):
                self.blockStep(mat,  lmd, lme, evec, w, w_copy, Nm, b,Nu)

            for u in range(Nu):
                for k in range(Nr):
                    lmd2[u,k]=lmd[u,k]
                    lme2[u,k]=lme[u,k]


            Qt = np.identity(Nr, ctype)
            
            # diagonalize
            t0 = g.time()
#            self.diagonalize(ev2, lme2, Nm, Qt)
            self.diagonalize(ev2,lmd2,lme2,Nu,Nr,Qt)
#    def diagonalize(self, eval, lmd, lme, Nu, Nk, Nm, Qt):
            t1 = g.time()

            if verbose:
                g.message("Diagonalization took %g s" % (t1 - t0))

            # sort
            ev2_copy = ev2.copy()
            ev2 = list(reversed(sorted(ev2)))

            for i in range(Nr):
#                if verbose:
                 g.message("Rval[%d]= %e"%(i,ev2[i]))

            # rotate
#            t0 = g.time()
#            g.rotate(evec, Qt, k1 - 1, k2 + 1, 0, Nm)
#            t1 = g.time()

#            if verbose:
#                g.message("Basis rotation took %g s" % (t1 - t0))

            # convergence test
            if it >= self.params["Nminres"]:
                if verbose:
                    g.message("Rotation to test convergence")

                # diagonalize
                for k in range(Nr):
                    ev2[k] = ev[k]
            #        lme2[k] = lme[k]
                for u in range(Nu):
                    for k in range(Nr):
                        lmd2[u,k]=lmd[u,k]
                        lme2[u,k]=lme[u,k]
                Qt = np.identity(Nm, ctype)

                t0 = g.time()
#                self.diagonalize(ev2, lme2, Nk, Qt)
                self.diagonalize(ev2,lmd2,lme2,Nu,Nr,Qt)
                t1 = g.time()

#                if verbose:
                g.message("Diagonalization took %g s" % (t1 - t0))

                B = g.copy(evec[0])

                allconv = True
#                if beta_k >= self.params["betastp"]:
                if  1 : 
                    jj = 1
                    while jj <= Nstop:
                        j = Nstop - jj
                        g.linear_combination(B, evec[0:Nr], Qt[j, 0:Nr])
                        if verbose:
                            g.message("norm=%e"%(g.norm2(B)))
                        B *= 1.0 / g.norm2(B) ** 0.5
                        if not ckpt.load(v):
                            mat(v, B)
                            ckpt.save(v)
                        ev_test = g.inner_product(B, v).real
                        eps2 = g.norm2(v - ev_test * B) / lambda_max ** 2.0
#                        if verbose:
                        if 1 :
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
#                    if verbose:
                    g.message("Converged in %d iterations" % it)
                    break

        t0 = g.time()
        g.rotate(evec, Qt, 0, Nstop, 0, Nk)
        t1 = g.time()

        if verbose:
            g.message("Final basis rotation took %g s" % (t1 - t0))

        return (evec[0:Nstop], ev2_copy[0:Nstop])

    def diagonalize(self, evals, lmd, lme, Nu, Nk, Qt):
        TriDiag = np.zeros((Nk, Nk), dtype=Qt.dtype)
        for u in range(Nu):
            for k in range(Nk):
                TriDiag[k][u+int(k/Nu)*Nu] = lmd[u][k]
        for u in range(Nu):
            for k in range(Nk):
                TriDiag[k-Nu][u+int(k/Nu)*Nu] = np.conjugate(lme[u][k-Nu])
                TriDiag[u+int(k/Nu)*Nu][k-Nu] = lme[u][k-Nu]
        w, v = np.linalg.eigh(TriDiag)
        for i in range(Nk):
            evals[Nk - 1 - i] = w[i]
            for j in range(Nk):
                Qt[Nk - 1 - i, j] = v[j,i]

    def blockStep(self, mat, lmd, lme, evec, w, w_copy, Nm, b, Nu):
        assert (b+1)*Nu <= Nm

        verbose = g.default.is_verbose("irl")
        ckpt = self.ckpt

        alph = 0.0
        beta = 0.0
        L= b*Nu
        R= (b+1)*Nu


        for k in range (L,R):
            if self.params["mem_report"]:
                g.mem_report(details=False)
# compute
            t0 = g.time()
            if not ckpt.load(w[k-L]):
                mat(w[k-L], evec[k])
#                            mat(v, B)
                ckpt.save(w[k-L])
            t1 = g.time()
    
                # allow to restrict maximal number of applications within run
            self.napply += 1
            if "maxapply" in self.params:
                if self.napply == self.params["maxapply"]:
                    if verbose:
                        g.message("Maximal number of matrix applications reached")
                    sys.exit(0)
        for u in range (Nu):
            for k in range (u,Nu):
                ip=g.inner_product(evec[L+k],evec[L+u])
                if np.abs(ip) >1e-6:
                    g.message("inner(evec[%d],evec[%d])=%e %e"% (L+k,L+u,ip.real,ip.imag))
    
        if b > 0:
            for u in range (Nu):
                for k in range (L-Nu+u,L):
                    w[u] -= np.conjugate(lme[u,k]) * evec[k]
                for k in range (L-Nu+u,L):
                    ip=g.inner_product(evec[k],w[u])
#                    if g.norm2(ip)>1e-6:
                    if np.abs(ip) >1e-6:
                        g.message("inner(evec[%d],w[%d])=%e %e"% (k,u,ip.real,ip.imag))
        else:
            for u in range (Nu):
                g.message("norm(evec[%d])=%e"%(u,g.norm2(evec[u])))

        for u in range (Nu):
            for k in range (L+u,R):
                lmd[u][k] = g.inner_product(evec[k],w[u])
                lmd[k-L][L+u]=np.conjugate(lmd[u][k])
            lmd[u][L+u]=np.real(lmd[u][L+u])

        for u in range (Nu):
            for k in range (L,R):
                w[u] -= lmd[u][k]*evec[k]
            for k in range (L,R):
                ip=g.inner_product(evec[k],w[u])
                if np.abs(ip) >1e-6:
                    g.message("inner(evec[%d],w[%d])=%e %e"% (k,u,ip.real,ip.imag))
            w_copy[u] = g.copy(w[u]);

        for u in range (Nu):
            for k in range (L,R):
                lme[u][k]=0.;
       
        for u in range (Nu):
            g.orthogonalize(w[u],evec[0:R])
            w[u] *= 1.0 / g.norm2(w[u]) ** 0.5
            for k in range (R):
                ip=g.inner_product(evec[k],w[u])
                if np.abs(ip) >1e-6:
                    g.message("inner(evec[%d],w[%d])=%e %e"% (k,u,ip.real,ip.imag))
       
        for u in range (Nu):
            g.orthogonalize(w[u],evec[0:R])
            w[u] *= 1.0 / g.norm2(w[u]) ** 0.5
            for k in range (R):
                ip=g.inner_product(evec[k],w[u])
                if np.abs(ip) >1e-6:
                    g.message("inner(evec[%d],w[%d])=%e %e"% (k,u,ip.real,ip.imag))

        for u in range (0,Nu):
            if u >0: 
                g.orthogonalize(w[u],w[0:u])
            w[u] *= 1.0 / g.norm2(w[u]) ** 0.5
            for k in range (u):
                ip=g.inner_product(w[k],w[u])
                if np.abs(ip) >1e-6:
                    g.message("inner(w[%d],w[%d])=%e %e"% (k,u,ip.real,ip.imag))
            ip=g.inner_product(w[u],w[u])
            g.message("inner(w[%d],w[%d])=%e %e"% (u,u,ip.real,ip.imag))

        for u in range (Nu):
            for v in range (u,Nu):
                lme[u][L+v] = g.inner_product(w[u],w_copy[v])
            lme[u][L+u] = np.real(lme[u][L+u])
        t3 = g.time()

        for u in range (Nu):
            for k in range (L+u,R):
                g.message( 
                    " In block %d, beta[%d][%d]=%e %e"
                    %( b, u, k-b*Nu,lme[u][k].real,lme[u][k].imag )
                )
    
#                ckpt.save([w, alph, beta])
    

        if b < (Nm/Nu - 1):
            for u in range (Nu):
                evec[R+u] = g.copy(w[u])
                ip=g.inner_product(evec[R+u],evec[R+u])
                if np.abs(ip) >1e-6:
                    g.message("inner(evec[%d],evec[%d])=%e %e"% (R+u,R+u,ip.real,ip.imag))
