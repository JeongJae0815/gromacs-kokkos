/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2012, The GROMACS development team.
 * Copyright (c) 2012,2013,2014,2015, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */

/*! \internal \file
 *  \brief
 *  Nonbonded Kokkos kernel functions.
 *
 *  \author Sikandar Y. Mashayak <symashayak@gmail.com>
 *  \ingroup module_mdlib
 */

#ifndef NBNXN_KOKKOS_KERNEL_H
#define NBNXN_KOKKOS_KERNEL_H

#include "gmxpre.h"

#include "config.h"

#include "gromacs/legacyheaders/force.h"
#include "gromacs/legacyheaders/gmx_omp_nthreads.h"
#include "gromacs/legacyheaders/typedefs.h"
#include "gromacs/legacyheaders/types/force_flags.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/nb_verlet.h"
#include "gromacs/mdlib/nbnxn_simd.h"
#include "gromacs/mdlib/nbnxn_kokkos.h"
#include "gromacs/mdlib/nbnxn_kernels/nbnxn_kernel_common.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/pbcutil/ishift.h"

#include "nbnxn_kokkos_types.h"

/*! \brief Kinds of electrostatic treatments in SIMD Verlet kernels
 */
enum {
    coulktRF, coulktTAB, coulktTAB_TWIN, coulktEWALD, coulktEWALD_TWIN, coulktNR
};

/*! \brief Kinds of Van der Waals treatments in SIMD Verlet kernels
 */
enum {
    vdwktLJCUT_COMBGEOM, vdwktLJCUT_COMBLB, vdwktLJCUT_COMBNONE, vdwktLJFORCESWITCH, vdwktLJPOTSWITCH, vdwktLJEWALDCOMBGEOM, vdwktNR
};

// transfer data from host-to-device if the data on the host is modified
void kokkos_sync_h2d (kokkos_atomdata_t*  kk_nbat,const kokkos_pairlist_t* kk_plist)
{

    kk_nbat->k_x.sync<GMXDeviceType>();
    kk_nbat->d_x = kk_nbat->k_x.d_view;
    kk_nbat->h_x = kk_nbat->k_x.h_view;

    // kk_plist->k_ci.sync<GMXDeviceType>();
    // kk_plist->k_sci.sync<GMXDeviceType>();
    // kk_plist->k_cj.sync<GMXDeviceType>();
    // kk_plist->k_cj4.sync<GMXDeviceType>();
    // kk_plist->k_excl.sync<GMXDeviceType>();


}

// transfer data from device-to-host if the data on the device is modified
void kokkos_sync_d2h (kokkos_atomdata_t*  kk_nbat,const kokkos_pairlist_t* kk_plist)
{

    //    kk_nbat->k_f.sync<GMXHosType>();

}

struct nbnxn_kokkos_kernel_functor
{
    // for now focusing on Xeon Phi device which is run in a native mode, i.e., host==device
    typedef GMXDeviceType device_type;

    // list of structures needed for non-bonded interactions
    const DAT::t_real_1d   x_;
    const HAT::t_un_real_1d q_;
    const HAT::t_un_ci_1d ci_;
    const HAT::t_un_cj_1d cj_;
    // atomic because more than one thread may write into same location
    const HAT::t_un_at_real_1d f_;

    const int nci_team_;

    // cluster size parameters
    // for now, Kokkos kernel uses 4*4 cluster size
    const int M_ = 4;
    const int N_ = 4;
    const int xstride_ = 3;
    const int fstride_ = 3;

    const real rcut2_;
    const real facel_;

    nbnxn_kokkos_kernel_functor (const DAT::t_real_1d x,
                                 const HAT::t_un_real_1d q,
                                 const HAT::t_un_ci_1d ci,
                                 const HAT::t_un_cj_1d cj,
                                 const HAT::t_un_real_1d f,
                                 const int nci_team,
                                 const real rcut2,
                                 const real facel):
        x_(x),q_(q),ci_(ci),cj_(cj),f_(f),
        nci_team_(nci_team),rcut2_(rcut2),facel_(facel)
    {

    };

    ~nbnxn_kokkos_kernel_functor ( )
    {

    };

    KOKKOS_FUNCTION
    real compute_item(const typename Kokkos::TeamPolicy<device_type>::member_type& dev) const
    {
        return 0.0;
    }

    // with thread teams
    KOKKOS_INLINE_FUNCTION
    void operator()(const  typename Kokkos::TeamPolicy<device_type>::member_type& dev) const
    {

        // print team information
        // if (dev.league_rank() == 0)
        // {
        //     printf("There are %d teams\n", dev.league_size());
        //     printf("There are %d threads per team\n", dev.team_size());
        // }

        //        printf("My team.thread id is %d . %d \n",dev.league_rank(),dev.team_rank());

        int ai, aj;
        int i, j;
        int cjind0, cjind1;
        int n, ci, ci_sh;
        int ish, ishf;
        int cj;

        real dx, dy, dz;
        real rsq, rinv;
        real rinvsq, rinvsix;
        real rcut2;
        real c6, c12;
        real FrLJ6 = 0, FrLJ12 = 0, frLJ = 0, VLJ = 0;
        real fscal;
        real fx, fy, fz;

        size_t x_size = M_*xstride_*sizeof(real);
        size_t f_size = M_*fstride_*sizeof(real);
        size_t q_size = M_*sizeof(real);

        real* xi_shmem = (real * ) dev.team_shmem().get_shmem(x_size);
        real* xj_shmem = (real * ) dev.team_shmem().get_shmem(x_size);
        real* fi_shmem = (real * ) dev.team_shmem().get_shmem(f_size);
        real* fj_shmem = (real * ) dev.team_shmem().get_shmem(f_size);
        real* qi_shmem = (real * ) dev.team_shmem().get_shmem(q_size);
        real* qj_shmem = (real * ) dev.team_shmem().get_shmem(q_size);

        const int ciind0 = dev.league_rank() * nci_team_;
        const int ciind1 = ciind0 + nci_team_;

        const int it = dev.team_rank();


        for (n = ciind0; n < ciind1; n++)
        {

            ish              = (ci_(n).shift & NBNXN_CI_SHIFT);
            /* x, f and fshift are assumed to be stored with stride 3 */
            ishf             = ish*DIM;
            cjind0           = ci_(n).cj_ind_start;
            cjind1           = ci_(n).cj_ind_end;
            /* Currently only works super-cells equal to sub-cells */
            ci               = ci_(n).ci;
            ci_sh            = (ish == CENTRAL ? ci : -1);

            // each thread loads its i atom from ci cluster into shared memory
            ai = ci * M_ + it;
            xi_shmem[it*xstride_ + XX] = x_(ai*xstride_ + XX);
            xi_shmem[it*xstride_ + YY] = x_(ai*xstride_ + YY);
            xi_shmem[it*xstride_ + ZZ] = x_(ai*xstride_ + ZZ);
            
            qi_shmem[it] = facel_ * q_(ai);
            
            fi_shmem[it*fstride_ + XX] = 0.0;
            fi_shmem[it*fstride_ + YY] = 0.0;
            fi_shmem[it*fstride_ + ZZ] = 0.0;

            int cjind = cjind0;
            while (cjind < cjind1 && cj_(cjind).excl != 0xffff)
            {
                cj = cj_(cjind).cj;
                aj = cj * N_ + it;

                fj_shmem[it*fstride_ + XX] = 0.0;
                fj_shmem[it*fstride_ + YY] = 0.0;
                fj_shmem[it*fstride_ + ZZ] = 0.0;
                
                xj_shmem[it*xstride_ + XX] = x_(aj*xstride_ + XX);
                xj_shmem[it*xstride_ + YY] = x_(aj*xstride_ + YY);
                xj_shmem[it*xstride_ + ZZ] = x_(aj*xstride_ + ZZ);

                qj_shmem[it] = q_(aj);

                //wait until all threads load their xi and xj
                dev.team_barrier();

                // each thread computes forces on its own i atom due to all j atoms in cj cluster
                // \todo this can be done using SIMD unit of each thread
                for (j = 0; j < N_; j++)
                {

                    /* A multiply mask used to zero an interaction
                     * when either the distance cutoff is exceeded, or
                     * (if appropriate) the i and j indices are
                     * unsuitable for this kind of inner loop. */
                    real skipmask;

#ifdef CHECK_EXCLS
                    /* A multiply mask used to zero an interaction
                     * when that interaction should be excluded
                     * (e.g. because of bonding). */
                    int interact;

                    interact = ((cj_(cjind).excl>>(it*M_ + j)) & 1);
#ifndef EXCL_FORCES
                    skipmask = interact;
#else
                    skipmask = !(cj == ci_sh && j <= it);
#endif
#else
#define interact 1.0
                    skipmask = 1.0;
#endif

                    dx = xi_shmem[it*xstride_ + XX] - xj_shmem[j*xstride_ + XX];
                    dy = xi_shmem[it*xstride_ + YY] - xj_shmem[j*xstride_ + YY];
                    dz = xi_shmem[it*xstride_ + ZZ] - xj_shmem[j*xstride_ + ZZ];
                    rsq = dx*dx + dy*dy + dz*dz;

                    /* Prepare to enforce the cut-off. */
                    skipmask = (rsq >= rcut2_) ? 0 : skipmask;
                    /* 9 flops for r^2 + cut-off check */

#ifdef CHECK_EXCLS
                    /* Excluded atoms are allowed to be on top of each other.
                     * To avoid overflow of rinv, rinvsq and rinvsix
                     * we add a small number to rsq for excluded pairs only.
                     */
                    rsq += (1 - interact)*NBNXN_AVOID_SING_R2_INC;
#endif

                    // \todo make sure gmx_invsqrt is callable from kokkos functor
                    rinv = gmx_invsqrt(rsq);
                    /* 5 flops for invsqrt */

                    /* Partially enforce the cut-off (and perhaps
                     * exclusions) to avoid possible overflow of
                     * rinvsix when computing LJ, and/or overflowing
                     * the Coulomb table during lookup. */
                    rinv = rinv * skipmask;

                    rinvsq  = rinv*rinv;

                    // \todo load c6 and c12 from nbfp
                    c6      = 0.26187E-02;   //nbfp[type_i_off+type[aj]*2  ];
                    c12     = 0.26307E-05;   //nbfp[type_i_off+type[aj]*2+1];

                    rinvsix = interact*rinvsq*rinvsq*rinvsq;
                    FrLJ6   = c6*rinvsix;
                    FrLJ12  = c12*rinvsix*rinvsix;
                    frLJ    = FrLJ12 - FrLJ6;

                    fscal = frLJ*rinvsq;

                    fx = fscal*dx;
                    fy = fscal*dy;
                    fz = fscal*dz;

                    /* Increment i-atom force */
                    fi_shmem[it*fstride_ + XX] += fx;
                    fi_shmem[it*fstride_ + YY] += fy;
                    fi_shmem[it*fstride_ + ZZ] += fz;

                    // /* Decrement j-atom force */
                    // do atomically: since all the threads write to fj_shmem
                    Kokkos::atomic_add(&fj_shmem[j*fstride_ + XX], -1.0*fx);
                    Kokkos::atomic_add(&fj_shmem[j*fstride_ + YY], -1.0*fy);
                    Kokkos::atomic_add(&fj_shmem[j*fstride_ + ZZ], -1.0*fz);
                    
                } // loop over j atoms

                //wait until all threads computed their forces btw i and j in current cj
                dev.team_barrier();

                f_(aj*fstride_ + XX) += fj_shmem[it*fstride_ + XX];
                f_(aj*fstride_ + YY) += fj_shmem[it*fstride_ + YY];
                f_(aj*fstride_ + ZZ) += fj_shmem[it*fstride_ + ZZ];

                dev.team_barrier();

                cjind++;

            } // loop over cj

            f_(ai*fstride_ + XX) += fi_shmem[it*fstride_ + XX];
            f_(ai*fstride_ + YY) += fi_shmem[it*fstride_ + YY];
            f_(ai*fstride_ + ZZ) += fi_shmem[it*fstride_ + ZZ];
            
        } // loop over ci

    }

    size_t team_shmem_size (int team_size) const {
        return sizeof(real ) * (M_* xstride_ + N_ * xstride_ + // xi + xj size
                                M_* fstride_ + N_ * fstride_ + // fi + fj size
                                M_ + N_ // qi + qj size
                                );
    }

};

void nbnxn_kokkos_launch_kernel(nbnxn_pairlist_set_t      *nbl_list,
                                const nbnxn_atomdata_t    *nbat,
                                const interaction_const_t *ic,
                                int                       ewald_excl,
                                rvec                      *shift_vec,
                                int                       force_flags,
                                int                       clearF,
                                real                      *fshift,
                                real                      *Vc,
                                real                      *Vvdw)
{

    int                nnbl;
    nbnxn_pairlist_t **nbl;
    int                nb;
    int                nthreads gmx_unused;

    nbnxn_atomdata_output_t *out;
    real                    *fshift_p;

    nnbl = nbl_list->nnbl;
    nbl  = nbl_list->nbl;

    out = &nbat->out[0];
    nb = 0; // single neighborlist for kokkos kernel
    if (clearF == enbvClearFYes)
    {
        clear_f(nbat, nb, out->f);
    }

    if ((force_flags & GMX_FORCE_VIRIAL) && nnbl == 1)
    {
        fshift_p = fshift;
    }
    else
    {
        fshift_p = out->fshift;

        if (clearF == enbvClearFYes)
        {
            clear_fshift(fshift_p);
        }
    }

    // initialize unmanaged f view from out->f
    nbat->kk_nbat->h_un_f =  HAT::t_un_real_1d(out->f, nbat->nalloc * nbat->fstride);

    // initialize unmanaged q view from nbat->
    nbat->kk_nbat->h_un_q = HAT::t_un_real_1d(nbat->q, nbat->nalloc);

    nthreads = gmx_omp_nthreads_get(emntNonbonded);

    // for now there is minimum 4 threads condition
    if (nthreads < 4)
    {
        gmx_incons("For nbnxn Kokkos kernel minimum 4 OpenMP thread required.");
    }

    // transfer data from host to device
    kokkos_sync_h2d(nbat->kk_nbat, nbl[0]->kk_plist);

    // initialize Kokkos functor

    const int teamsize = 4; //nbl[0]->na_ci;
    const int nteams = int(nthreads/teamsize);
    const int nci_team = int(nbl[0]->nci/nteams) + 1;

    typedef nbnxn_kokkos_kernel_functor f_type;
    f_type nb_f(nbat->kk_nbat->d_x,
                nbat->kk_nbat->h_un_q,
                nbl[0]->kk_plist->h_un_ci,
                nbl[0]->kk_plist->h_un_cj,
                nbat->kk_nbat->h_un_f,
                nci_team,
                ic->rcoulomb*ic->rcoulomb,
                ic->epsfac);

    //    Kokkos::DefaultExecutionSpace::print_configuration(std::cout,true);
    Kokkos::TeamPolicy<typename f_type::device_type> config(nteams,teamsize);

    // printf("\n number of threads per team %d\n", teamsize);
    // printf("\n number of teams %d\n", nteams);
    // printf("\n number of i clusters %d\n", nbl[0]->nci);

    // if (!(force_flags & GMX_FORCE_ENERGY))
    // {
        /* Don't calculate energies */
        //     p_nbk_noener[coulkt][vdwkt](nbl[nb], nbat,
        //                                 ic,
        //                                 shift_vec,
        //                                 out->f,
        //                                 fshift_p);

        // launch kernel
        Kokkos::parallel_for(config,nb_f);
    // }
    // else
    // {
    //     gmx_incons("nbnxn Kokkos kernel doesn't yet support energy calculations.");
    // }

        // deallocate unmanaged views
        nbat->kk_nbat->h_un_f =  HAT::t_un_real_1d();
        nbat->kk_nbat->h_un_q = HAT::t_un_real_1d();


}

#endif


// // following loop structure based on clusters explained in fig. 6
// // of ref. S. Pall and B. Hess, Comp. Phys. Comm., 184, 2013
// // for each cj cluster
// //   load N coords+params for cj
// //   for j = 0 to M
// //     for i = 0 to N
// //       calculate interaction of ci*M+i  with cj*N+j
// //   store N cj-forces
// // store M ci-forces
