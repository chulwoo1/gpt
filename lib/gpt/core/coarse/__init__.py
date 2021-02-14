#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Daniel Richtmann (daniel.richtmann@ur.de)
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
import gpt, cgpt, numpy


#
# rhs_n_block: How many vectors are acted on at the same time
#
@gpt.params.params_convention(make_hermitian=False, save_links=True, rhs_n_block=4)
def create_links(A, fmat, basis, params):
    # NOTE: we expect the blocks in the basis vectors
    # to already be orthogonalized!
    # parameters
    make_hermitian = params["make_hermitian"]
    save_links = params["save_links"]
    rhs_n_block = params["rhs_n_block"]
    assert not (make_hermitian and not save_links)

    # sanity
    rhs_n_block = min(len(basis), rhs_n_block)

    # verbosity
    verbose = gpt.default.is_verbose("coarsen")

    # setup timings
    t = gpt.timer("coarsen")
    t("setup")

    # get grids
    f_grid = basis[0].grid
    c_grid = A[0].grid

    # directions/displacements we coarsen for
    assert f_grid.nd in [
        4,
        5,
    ]  # TODO: this is poor design, need to make this independent of ndim
    dirs = [1, 2, 3, 4] if f_grid.nd == 5 else [0, 1, 2, 3]
    disp = +1
    dirdisps_full = list(zip(dirs * 2, [+1] * 4 + [-1] * 4))
    dirdisps_forward = list(zip(dirs, [disp] * 4))
    nhops = len(dirdisps_full)
    selflink = nhops

    # setup fields
    Mbasis = [[gpt.lattice(basis[0]) for j in range(rhs_n_block)] for i in range(nhops)]
    tmp = [gpt.lattice(basis[0]) for j in range(rhs_n_block)]
    oproj = [gpt.vcomplex(c_grid, len(basis)) for j in range(rhs_n_block)]
    selfproj = [gpt.vcomplex(c_grid, len(basis)) for j in range(rhs_n_block)]

    # setup masks
    onemask, blockevenmask, blockoddmask = (
        gpt.complex(f_grid),
        gpt.complex(f_grid),
        gpt.complex(f_grid),
    )
    dirmasks = [gpt.complex(f_grid) for p in range(nhops)]

    # auxilliary stuff needed for masks
    t("masks")
    onemask[:] = 1.0
    coor = gpt.coordinates(blockevenmask)
    block = numpy.array(f_grid.ldimensions) / numpy.array(c_grid.ldimensions)
    block_cb = coor[:, :] // block[:]

    # fill masks for sites within even/odd blocks
    gpt.coordinate_mask(blockevenmask, numpy.sum(block_cb, axis=1) % 2 == 0)
    blockoddmask @= onemask - blockevenmask

    # fill masks for sites on borders of blocks
    dirmasks_forward_np = coor[:, :] % block[:] == block[:] - 1
    dirmasks_backward_np = coor[:, :] % block[:] == 0
    for mu in dirs:
        gpt.coordinate_mask(dirmasks[mu], dirmasks_forward_np[:, mu])
        gpt.coordinate_mask(dirmasks[mu + 4], dirmasks_backward_np[:, mu])

    # save applications of matrix and coarsening if possible
    dirdisps = dirdisps_forward if save_links else dirdisps_full

    # create block maps
    t("blockmap")
    dirbms = [
        gpt.block.map(c_grid, basis, dirmasks[p]) for p, (mu, fb) in enumerate(dirdisps)
    ]
    fullbm = gpt.block.map(c_grid, basis)

    for i0 in range(0, len(basis), rhs_n_block):
        # rhs indices
        t("indices")
        i1 = min(i0 + rhs_n_block, len(basis))
        iblock = i1 - i0

        # apply directional hopping terms
        # this triggers len(dirdisps) comms -> TODO expose DhopdirAll from Grid
        # BUT problem with vector<Lattice<...>> in rhs
        t("apply_hop")
        for p, dirdisp in enumerate(dirdisps):
            fmat.Mdir(*dirdisp)(Mbasis[p][0:iblock], basis[i0:i1])

        # coarsen directional terms + write to link
        for p, (mu, fb) in enumerate(dirdisps):
            t("coarsen_hop")
            dirbms[p].project(oproj, Mbasis[p])

            t("copy_hop")
            for ib in range(iblock):
                A[p][:, :, :, :, :, i0 + ib] = oproj[ib][:]

        # fast diagonal term: apply full matrix to both block cbs separately and discard hops into other cb
        t("apply_self")
        for ib in range(iblock):
            tmp[ib] @= (
                blockevenmask * fmat * basis[i0 + ib] * blockevenmask
                + blockoddmask * fmat * basis[i0 + ib] * blockoddmask
            )

        # coarsen diagonal term
        t("coarsen_self")
        fullbm.project(selfproj, tmp)

        # write to self link
        t("copy_self")
        for ib in range(iblock):
            A[selflink][:, :, :, :, :, i0 + ib] = selfproj[ib][:]

        if verbose:
            gpt.message("coarsen: done with vectors %3d - %3d" % (i0, i1 - 1))

    # communicate opposite links
    if save_links:
        t("comm")
        communicate_links(A, dirdisps_forward, make_hermitian)

    t()

    if verbose:
        gpt.message(t)


def communicate_links(A, dirdisps_forward, make_hermitian):
    assert type(A) == list
    assert len(A) == 2 * len(dirdisps_forward) + 1
    for p, (mu, fb) in enumerate(dirdisps_forward):
        p_other = p + 4
        shift_fb = fb * -1
        Atmp = gpt.copy(A[p])
        if not make_hermitian:
            # Atmp = prefactor_dagger(Atmp) * Atmp  # this would be more elegant
            nbasis = A[0].otype.shape[0]
            assert nbasis % 2 == 0
            nb = nbasis // 2
            Atmp[:, :, :, :, 0:nb, nb:nbasis] *= -1.0  # upper right block
            Atmp[:, :, :, :, nb:nbasis, 0:nb] *= -1.0  # lower left block
        A[p_other] @= gpt.adj(gpt.cshift(Atmp, mu, shift_fb))
