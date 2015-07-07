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

#include "gmxpre.h"

#include "config.h"

#include "gromacs/legacyheaders/force.h"
#include "gromacs/legacyheaders/gmx_omp_nthreads.h"
#include "gromacs/legacyheaders/typedefs.h"
#include "gromacs/legacyheaders/types/force_flags.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/nbnxn_consts.h"
#include "gromacs/mdlib/nb_verlet.h"
#include "gromacs/mdlib/nbnxn_simd.h"
#include "gromacs/mdlib/nbnxn_kokkos.h"
#include "gromacs/mdlib/nbnxn_kernels/nbnxn_kernel_common.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/pbcutil/ishift.h"

#include "nbnxn_kokkos_types.h"

/* We could use nbat->xstride and nbat->fstride, but macros might be faster */
#define X_STRIDE   3
#define F_STRIDE   3

/* For Kokkos, for now, using cluster sizes same as CPU i.e. 4 */
#define NBNXN_KOKKOS_CLUSTER_I_SIZE NBNXN_CPU_CLUSTER_I_SIZE // 4
#define NBNXN_KOKKOS_CLUSTER_J_SIZE NBNXN_CPU_CLUSTER_I_SIZE // 4

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
    //    const DAT::t_real_1d   x_;
    HAT::t_un_real_1d x_;
    HAT::t_un_real_1d q_;
    HAT::t_un_int_1d type_;
    HAT::t_un_real_1d nbfp_;
    HAT::t_un_ci_1d ci_;
    HAT::t_un_cj_1d cj_;
    // atomic because more than one thread may write into same location
    HAT::t_un_at_real_1d f_;
    HAT::t_un_real_1d Ftab_;
    HAT::t_un_real_1d shiftvec_;

    const int nci_team_;
    const real rcut2_, facel_, tabq_scale_;
    const int ntype_;

    nbnxn_kokkos_kernel_functor (const HAT::t_un_real_1d &x,
                                 const HAT::t_un_real_1d &q,
                                 const HAT::t_un_int_1d &type,
                                 const HAT::t_un_real_1d &nbfp,
                                 const HAT::t_un_ci_1d &ci,
                                 const HAT::t_un_cj_1d &cj,
                                 const HAT::t_un_real_1d &f,
                                 const HAT::t_un_real_1d &Ftab,
                                 const HAT::t_un_real_1d &shiftvec,
                                 const int &nci_team,
                                 const real &rcut2,
                                 const real &facel,
                                 const real &tabq_scale,
                                 const int &ntype):
        x_(x),q_(q),type_(type),nbfp_(nbfp),ci_(ci),cj_(cj),f_(f),Ftab_(Ftab),shiftvec_(shiftvec),
        nci_team_(nci_team),rcut2_(rcut2),facel_(facel),tabq_scale_(tabq_scale),ntype_(ntype)
    {

    };

    ~nbnxn_kokkos_kernel_functor ( )
    {

    };

    // with thread teams
    KOKKOS_INLINE_FUNCTION
    void operator()(const  typename Kokkos::TeamPolicy<device_type>::member_type& dev) const
    {

        //        print team information
        // if (dev.league_rank() == 0)
        // {
        //     printf("There are %d teams\n", dev.league_size());
        //     printf("There are %d threads per team\n", dev.team_size());
        // }

        // printf("My team.thread id is %d . %d \n",dev.league_rank(),dev.team_rank());

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
        real fx_total, fy_total, fz_total;
        int i_type, j_type;
        int j_global;

        real qq;
        real fcoul;
        real rs, frac;
        int  ri;
        real fexcl;

        size_t x_size = NBNXN_KOKKOS_CLUSTER_I_SIZE*X_STRIDE*sizeof(real);
        size_t f_size = NBNXN_KOKKOS_CLUSTER_I_SIZE*F_STRIDE*sizeof(real);
        size_t q_size = NBNXN_KOKKOS_CLUSTER_I_SIZE*sizeof(real);

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

            // \todo here NBNXN_CI_SHIFT=127, find out is this the right multiplier for kokkos kernel
            ish              = (ci_(n).shift & NBNXN_CI_SHIFT);
            /* x, f and fshift are assumed to be stored with stride 3 */
            ishf             = ish*DIM;
            cjind0           = ci_(n).cj_ind_start;
            cjind1           = ci_(n).cj_ind_end;
            /* Currently only works super-cells equal to sub-cells */
            ci               = ci_(n).ci;
            ci_sh            = (ish == CENTRAL ? ci : -1);

            // each thread loads its i atom from ci cluster into shared memory
            ai = ci * NBNXN_KOKKOS_CLUSTER_I_SIZE + it;
            i_type = ntype_*2*type_(ai);
            xi_shmem[it*X_STRIDE + XX] = x_(ai*X_STRIDE + XX) + shiftvec_(ishf + XX);
            xi_shmem[it*X_STRIDE + YY] = x_(ai*X_STRIDE + YY) + shiftvec_(ishf + YY);
            xi_shmem[it*X_STRIDE + ZZ] = x_(ai*X_STRIDE + ZZ) + shiftvec_(ishf + ZZ);
            
            qi_shmem[it] = facel_ * q_(ai);
            
            fi_shmem[it*F_STRIDE + XX] = 0.0;
            fi_shmem[it*F_STRIDE + YY] = 0.0;
            fi_shmem[it*F_STRIDE + ZZ] = 0.0;

            int cjind = cjind0;
            while (cjind < cjind1 && cj_(cjind).excl != 0xffff)
            {

#define CHECK_EXCLS

#include "gromacs/mdlib/nbnxn_kokkos/nbnxn_kokkos_kernel_inner.h"

#undef CHECK_EXCLS

                cjind++;

            } // loop over cj

            for (; (cjind < cjind1); cjind++)
            {

#include "gromacs/mdlib/nbnxn_kokkos/nbnxn_kokkos_kernel_inner.h"

            }
            
            // atomic addition
            f_(ai*F_STRIDE + XX) += fi_shmem[it*F_STRIDE + XX];
            f_(ai*F_STRIDE + YY) += fi_shmem[it*F_STRIDE + YY];
            f_(ai*F_STRIDE + ZZ) += fi_shmem[it*F_STRIDE + ZZ];

            // \todo add i forces to shift force array

        } // loop over ci

    }

    size_t team_shmem_size (int team_size) const {
        return sizeof(real ) * (NBNXN_KOKKOS_CLUSTER_I_SIZE * X_STRIDE + NBNXN_KOKKOS_CLUSTER_J_SIZE * X_STRIDE + // xi + xj size
                                NBNXN_KOKKOS_CLUSTER_I_SIZE * F_STRIDE + NBNXN_KOKKOS_CLUSTER_J_SIZE * F_STRIDE + // fi + fj size
                                NBNXN_KOKKOS_CLUSTER_I_SIZE + NBNXN_KOKKOS_CLUSTER_J_SIZE // qi + qj size
                                );
    }

};

void nbnxn_kokkos_launch_kernel(nbnxn_pairlist_set_t      *nbl_list,
                                nbnxn_atomdata_t    *nbat,
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

    // printf("Number of neighborlist = %d", nnbl);

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

    // initialize unmanaged Kokkos host views from existing arrays on the host

    if (nbat->kk_nbat == NULL)
    {
        snew(nbat->kk_nbat,1);    
    }

    nbat->kk_nbat->h_un_x = HAT::t_un_real_1d(nbat->x, nbat->nalloc * nbat->xstride);
    nbat->kk_nbat->h_un_q = HAT::t_un_real_1d(nbat->q, nbat->nalloc);
    nbat->kk_nbat->h_un_f = HAT::t_un_real_1d(out->f, nbat->nalloc * nbat->fstride);

    nbat->kk_nbat->h_un_type = HAT::t_un_int_1d(nbat->type, nbat->ntype);
    nbat->kk_nbat->h_un_nbfp = HAT::t_un_real_1d(nbat->nbfp, nbat->ntype*nbat->ntype*2);
    nbat->kk_nbat->h_un_Ftab = HAT::t_un_real_1d(ic->tabq_coul_F, ic->tabq_size);

    nbat->kk_nbat->h_un_shiftvec = HAT::t_un_real_1d(shift_vec[0], SHIFTS);

    if (nbl[0]->kk_plist == NULL)
    {
        snew(nbl[0]->kk_plist,1);    
    }

    nbl[0]->kk_plist->h_un_ci = HAT::t_un_ci_1d(nbl[0]->ci,nbl[0]->ci_nalloc);
    nbl[0]->kk_plist->h_un_cj = HAT::t_un_cj_1d(nbl[0]->cj,nbl[0]->cj_nalloc);

    nthreads = gmx_omp_nthreads_get(emntNonbonded);

    // for now there is minimum 4 threads condition
    if (nthreads < 4)
    {
        gmx_incons("For nbnxn Kokkos kernel minimum 4 OpenMP thread required.");
    }

    // transfer data from host to device
    // kokkos_sync_h2d(nbat->kk_nbat, nbl[0]->kk_plist);

    // initialize Kokkos functor

    const int teamsize = 4; //nbl[0]->na_ci;
    const int nteams = int(nthreads/teamsize);
    const int nci_team = int(nbl[0]->nci/nteams) + 1;

    typedef nbnxn_kokkos_kernel_functor f_type;
    f_type nb_f(nbat->kk_nbat->h_un_x,
                nbat->kk_nbat->h_un_q,
                nbat->kk_nbat->h_un_type,
                nbat->kk_nbat->h_un_nbfp,
                nbl[0]->kk_plist->h_un_ci,
                nbl[0]->kk_plist->h_un_cj,
                nbat->kk_nbat->h_un_f,
                nbat->kk_nbat->h_un_Ftab,
                nbat->kk_nbat->h_un_shiftvec,
                nci_team,
                ic->rcoulomb*ic->rcoulomb,
                ic->epsfac,
                ic->tabq_scale,
                nbat->ntype);

    // Kokkos::DefaultExecutionSpace::print_configuration(std::cout,true);
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
    // for now, wait until parallel_for finishes
    Kokkos::fence();
    // }
    // else
    // {
    //     gmx_incons("nbnxn Kokkos kernel doesn't yet support energy calculations.");
    // }

    // // print forces
    // printf("Total forces\n");
    // int i = 1000;
    // // for ( i = 0; i < nbat->nalloc; i++)
    // {
    //     printf("i = %d, fx = %lf\n", i, out->f[i*F_STRIDE+XX]);
    //     printf("i = %d, fy = %lf\n", i, out->f[i*F_STRIDE+YY]);
    //     printf("i = %d, fz = %lf\n", i, out->f[i*F_STRIDE+ZZ]);
    // }
    // deallocate unmanaged views
    nbat->kk_nbat->h_un_f =  HAT::t_un_real_1d();
    nbat->kk_nbat->h_un_q = HAT::t_un_real_1d();


}

#undef X_STRIDE
#undef F_STRIDE
#undef NBNXN_KOKKOS_CLUSTER_I_SIZE
#undef NBNXN_KOKKOS_CLUSTER_J_SIZE
// // following loop structure based on clusters explained in fig. 6
// // of ref. S. Pall and B. Hess, Comp. Phys. Comm., 184, 2013
// // for each cj cluster
// //   load N coords+params for cj
// //   for j = 0 to M
// //     for i = 0 to N
// //       calculate interaction of ci*M+i  with cj*N+j
// //   store N cj-forces
// // store M ci-forces
