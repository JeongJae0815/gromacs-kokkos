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

#include <time.h>

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
#define XI_STRIDE   3
#define FI_STRIDE   3

/* For Kokkos, for now, using cluster sizes same as CPU i.e. 4 */
#define NBNXN_KOKKOS_CLUSTER_I_SIZE NBNXN_CPU_CLUSTER_I_SIZE // 4
#define NBNXN_KOKKOS_CLUSTER_J_SIZE NBNXN_CPU_CLUSTER_I_SIZE // 4

#define UNROLLI NBNXN_KOKKOS_CLUSTER_I_SIZE
#define UNROLLJ NBNXN_KOKKOS_CLUSTER_J_SIZE

struct nbnxn_kokkos_kernel_functor
{
    // for now focusing on Xeon Phi device which is run in a native mode, i.e., host==device
    typedef GMXDeviceType device_type;

    // list of structures needed for non-bonded interactions
    HAT::t_un_real_1d x_;
    HAT::t_un_real_1d q_;
    HAT::t_un_int_1d type_;
    HAT::t_un_real_1d nbfp_;
    HAT::t_un_ci_1d* ci_;
    HAT::t_un_cj_1d* cj_;
    // make this atomic when vectorizing because more than one thread may write into same location
    HAT::t_un_real_1d* f_;
    HAT::t_un_real_1d* Vvdw_;
    HAT::t_un_real_1d* Vc_;
    HAT::t_un_real_1d Ftab_;
    HAT::t_un_real_1d Vtab_;
    HAT::t_un_real_1d shiftvec_;

    typedef Kokkos::
    View<int*, Kokkos::LayoutRight, GMXHostType> t_int_1d;
    t_int_1d nci_;

    const real rcut2_, facel_, tabq_scale_;
    const real repulsion_shift_cpot_, dispersion_shift_cpot_;
    const real sh_ewald_;
    const int ntype_;
    const int nnbl_;

    nbnxn_kokkos_kernel_functor (const HAT::t_un_real_1d &Ftab,
                                 const HAT::t_un_real_1d &Vtab,
                                 const HAT::t_un_real_1d &shiftvec,
                                 const real &rcut2,
                                 const real &facel,
                                 const real &tabq_scale,
                                 const real &repulsion_shift_cpot,
                                 const real &dispersion_shift_cpot,
                                 const real &sh_ewald,
                                 const int &ntype,
                                 nbnxn_atomdata_t    *nbat,                                 
                                 nbnxn_pairlist_t **nbl,
                                 const int &nnbl):
        Ftab_(Ftab),Vtab_(Vtab),shiftvec_(shiftvec),
        rcut2_(rcut2),facel_(facel),tabq_scale_(tabq_scale),
        repulsion_shift_cpot_(repulsion_shift_cpot),dispersion_shift_cpot_(dispersion_shift_cpot),
        sh_ewald_(sh_ewald),ntype_(ntype),nnbl_(nnbl)
    {
        int nb;

        x_    = HAT::t_un_real_1d(nbat->x, nbat->nalloc * nbat->xstride);
        q_    = HAT::t_un_real_1d(nbat->q, nbat->nalloc);
        type_ = HAT::t_un_int_1d(nbat->type, nbat->ntype);
        nbfp_ = HAT::t_un_real_1d(nbat->nbfp, nbat->ntype*nbat->ntype*2);

        snew(ci_,nnbl_);
        snew(cj_,nnbl_);
        nci_ = t_int_1d("nci",nnbl_);

        snew(f_,nnbl_);
        snew(Vvdw_,nnbl_);
        snew(Vc_,nnbl_);

        for (nb = 0; nb < nnbl_; nb++)
        {
            ci_[nb]   = HAT::t_un_ci_1d(nbl[nb]->ci,nbl[nb]->ci_nalloc);
            cj_[nb]   = HAT::t_un_cj_1d(nbl[nb]->cj,nbl[nb]->cj_nalloc);
            nci_(nb)  = nbl[nb]->nci;
            f_[nb]    = HAT::t_un_real_1d(nbat->out[nb].f, nbat->nalloc * nbat->fstride);
            Vvdw_[nb] = HAT::t_un_real_1d(nbat->out[nb].Vvdw, nbat->out[nb].nV);
            Vc_[nb]   = HAT::t_un_real_1d(nbat->out[nb].Vc, nbat->out[nb].nV);
        }
        
    };

    ~nbnxn_kokkos_kernel_functor ( )
    {

    };

    // no vectorization
    KOKKOS_INLINE_FUNCTION
    void operator()(const int I) const
    {


        real                facel;
        real               *nbfp_i;
        int                 n, ci, ci_sh;
        int                 ish, ishf;
        gmx_bool            do_LJ, half_LJ, do_coul, do_self;
        int                 cjind0, cjind1, cjind;
        int                 ip, jp;

        real                xi[UNROLLI*XI_STRIDE];
        real                fi[UNROLLI*FI_STRIDE];
        real                qi[UNROLLI];

        real                Vvdw_ci, Vc_ci;

        const real          tabscale = tabq_scale_;
        const real          halfsp = 0.5/tabscale;

        const real          rcut2 = rcut2_;
        const int           ntype2 = ntype_*2;

        int ninner;

        Vvdw_[I](0) = 0.0;
        Vc_[I](0) = 0.0;

        for (n = 0; n < nci_(I); n++)
        {
            int i, d;

            ish              = (ci_[I](n).shift & NBNXN_CI_SHIFT);
            /* x, f and fshift are assumed to be stored with stride 3 */
            ishf             = ish*DIM;
            cjind0           = ci_[I](n).cj_ind_start;
            cjind1           = ci_[I](n).cj_ind_end;
            /* Currently only works super-cells equal to sub-cells */
            ci               = ci_[I](n).ci;
            ci_sh            = (ish == CENTRAL ? ci : -1);

            /* We have 5 LJ/C combinations, but use only three inner loops,
             * as the other combinations are unlikely and/or not much faster:
             * inner half-LJ + C for half-LJ + C / no-LJ + C
             * inner LJ + C      for full-LJ + C
             * inner LJ          for full-LJ + no-C / half-LJ + no-C
             */
            do_LJ   = (ci_[I](n).shift & NBNXN_CI_DO_LJ(0));
            do_coul = (ci_[I](n).shift & NBNXN_CI_DO_COUL(0));
            half_LJ = ((ci_[I](n).shift & NBNXN_CI_HALF_LJ(0)) || !do_LJ) && do_coul;
            do_self = do_coul;

            Vvdw_ci = 0;
            Vc_ci   = 0;

            for (i = 0; i < UNROLLI; i++)
            {
                int ai = ci * NBNXN_KOKKOS_CLUSTER_I_SIZE + i;
                xi[i*XI_STRIDE + XX] = x_(ai*XI_STRIDE + XX) + shiftvec_(ishf + XX);
                xi[i*XI_STRIDE + YY] = x_(ai*XI_STRIDE + YY) + shiftvec_(ishf + YY);
                xi[i*XI_STRIDE + ZZ] = x_(ai*XI_STRIDE + ZZ) + shiftvec_(ishf + ZZ);
                qi[i] = facel_ * q_(ai);
                fi[i*FI_STRIDE + XX] = 0.0;
                fi[i*FI_STRIDE + YY] = 0.0;
                fi[i*FI_STRIDE + ZZ] = 0.0;
            }

            if (do_self)
            {
                real Vc_sub_self = 0.5*Vtab_(0);

                if (cj_[I](cjind0).cj == ci_sh)
                {
                    for (i = 0; i < UNROLLI; i++)
                    {
                        int egp_ind = 0;
                        /* Coulomb self interaction */
                        Vc_[I](egp_ind)   -= qi[i]*q_(ci*UNROLLI+i)*Vc_sub_self;
                    }
                }
            }

            cjind = cjind0;

            while (cjind < cjind1 && cj_[I](cjind).excl != 0xffff)
            {
#define CHECK_EXCLS
                if (half_LJ)
                {
#define CALC_COULOMB
#define HALF_LJ
#include "gromacs/mdlib/nbnxn_kokkos/nbnxn_kokkos_kernel_inner_omp.h"
#undef HALF_LJ
#undef CALC_COULOMB
                }
                else if (do_coul)
                {
#define CALC_COULOMB
#include "gromacs/mdlib/nbnxn_kokkos/nbnxn_kokkos_kernel_inner_omp.h"
#undef CALC_COULOMB
                }
                else
                {
#include "gromacs/mdlib/nbnxn_kokkos/nbnxn_kokkos_kernel_inner_omp.h"
                }
#undef CHECK_EXCLS
                cjind++;
            } 

            for (; (cjind < cjind1); cjind++)
            {
                if (half_LJ)
                {
#define CALC_COULOMB
#define HALF_LJ
#include "gromacs/mdlib/nbnxn_kokkos/nbnxn_kokkos_kernel_inner_omp.h"
#undef HALF_LJ
#undef CALC_COULOMB
                }
                else if (do_coul)
                {
#define CALC_COULOMB
#include "gromacs/mdlib/nbnxn_kokkos/nbnxn_kokkos_kernel_inner_omp.h"
#undef CALC_COULOMB
                }
                else
                {
#include "gromacs/mdlib/nbnxn_kokkos/nbnxn_kokkos_kernel_inner_omp.h"
                }
            }
            ninner += cjind1 - cjind0;

            /* Add accumulated i-forces to the force array */
            for (i = 0; i < UNROLLI; i++)
            {
                for (d = 0; d < DIM; d++)
                {
                    f_[I]((ci*UNROLLI+i)*FI_STRIDE+d) += fi[i*FI_STRIDE+d];
                }
            }

            Vvdw_[I](0) += Vvdw_ci;
            Vc_[I](0) += Vc_ci;

        }

    } // operator()

    // with vectorization
    KOKKOS_INLINE_FUNCTION
    void operator()(const  typename Kokkos::TeamPolicy<device_type>::member_type& dev) const
    {
        int n;
        int cjind0, cjind1;
        int ci, ci_sh;
        int ish, ishf;
        int cj;

        const int I = dev.league_rank() * dev.team_size() + dev.team_rank();

        size_t x_size  = NBNXN_KOKKOS_CLUSTER_I_SIZE*XI_STRIDE*sizeof(real);
        real* xi_shmem = (real * ) dev.team_shmem().get_shmem(x_size);
        real* xj_shmem = (real * ) dev.team_shmem().get_shmem(x_size);

        size_t q_size  = NBNXN_KOKKOS_CLUSTER_I_SIZE*sizeof(real);
        real* qi_shmem = (real * ) dev.team_shmem().get_shmem(q_size);
        real* qj_shmem = (real * ) dev.team_shmem().get_shmem(q_size);

        size_t f_size  = NBNXN_KOKKOS_CLUSTER_I_SIZE*FI_STRIDE*sizeof(real);
        real* fi_shmem = (real * ) dev.team_shmem().get_shmem(f_size);

        struct fj_reduce
        {
            real f[NBNXN_KOKKOS_CLUSTER_J_SIZE*FI_STRIDE];

            KOKKOS_INLINE_FUNCTION
            void operator+=(const volatile struct fj_reduce &rhs) volatile
            {

                for (int j = 0; j < NBNXN_KOKKOS_CLUSTER_J_SIZE*FI_STRIDE; j++)
                {
                    f[j] += rhs.f[j];
                }
            }

        };

        struct fj_reduce fj_sh;

        for (n = 0; n < nci_(I); n++)
        {
            // \todo here NBNXN_CI_SHIFT=127, find out is this the right multiplier for kokkos kernel
            ish              = (ci_[I](n).shift & NBNXN_CI_SHIFT);
            /* x, f and fshift are assumed to be stored with stride 3 */
            ishf             = ish*DIM;
            cjind0           = ci_[I](n).cj_ind_start;
            cjind1           = ci_[I](n).cj_ind_end;
            /* Currently only works super-cells equal to sub-cells */
            ci               = ci_[I](n).ci;
            ci_sh            = (ish == CENTRAL ? ci : -1);

            // load cluster i coordinates into shared memory
            Kokkos::parallel_for
                (Kokkos::ThreadVectorRange(dev,NBNXN_KOKKOS_CLUSTER_I_SIZE), [&] (const int& k)
                 {
                     int ai = ci * NBNXN_KOKKOS_CLUSTER_I_SIZE + k;
                     xi_shmem[k*XI_STRIDE + XX] = x_(ai*XI_STRIDE + XX) + shiftvec_(ishf + XX);
                     xi_shmem[k*XI_STRIDE + YY] = x_(ai*XI_STRIDE + YY) + shiftvec_(ishf + YY);
                     xi_shmem[k*XI_STRIDE + ZZ] = x_(ai*XI_STRIDE + ZZ) + shiftvec_(ishf + ZZ);
                     qi_shmem[k] = facel_ * q_(ai);
                     fi_shmem[k*FI_STRIDE + XX] = 0.0;
                     fi_shmem[k*FI_STRIDE + YY] = 0.0;
                     fi_shmem[k*FI_STRIDE + ZZ] = 0.0;
                 });

            int cjind = cjind0;

#define CHECK_EXCLS

            while (cjind < cjind1 && cj_[I](cjind).excl != 0xffff)
            {
#include "gromacs/mdlib/nbnxn_kokkos/nbnxn_kokkos_kernel_inner.h"
                cjind++;
            } 

#undef CHECK_EXCLS

            for (; (cjind < cjind1); cjind++)
            {
#include "gromacs/mdlib/nbnxn_kokkos/nbnxn_kokkos_kernel_inner.h"
            }

                
            // add i forces to global f
            // atomic addition
            Kokkos::parallel_for
                (Kokkos::ThreadVectorRange(dev,NBNXN_KOKKOS_CLUSTER_I_SIZE), [&] (const int& k)
                 {
                     int ai = ci * NBNXN_KOKKOS_CLUSTER_I_SIZE + k;
                     f_[I](ai*FI_STRIDE + XX) += fi_shmem[k*FI_STRIDE + XX];
                     f_[I](ai*FI_STRIDE + YY) += fi_shmem[k*FI_STRIDE + YY];
                     f_[I](ai*FI_STRIDE + ZZ) += fi_shmem[k*FI_STRIDE + ZZ];
                 });




        } // for loop over ci
     
    } // operator()
        
    size_t team_shmem_size (int team_size) const {
        return sizeof(real ) * (NBNXN_KOKKOS_CLUSTER_I_SIZE * XI_STRIDE + NBNXN_KOKKOS_CLUSTER_J_SIZE * XI_STRIDE + // xi + xj size
                                NBNXN_KOKKOS_CLUSTER_I_SIZE * FI_STRIDE + NBNXN_KOKKOS_CLUSTER_J_SIZE * FI_STRIDE + // fi + fj size
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

    // for now using omp pragma to clear forces
    // \todo implement this as kokkos parallel_for
    nthreads = gmx_omp_nthreads_get(emntNonbonded);
#pragma omp parallel for schedule(static) num_threads(nthreads)
    for (nb = 0; nb < nnbl; nb++)
    {
        nbnxn_atomdata_output_t *out;
        real                    *fshift_p;

        out = &nbat->out[nb];

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
    }

    // initialize unmanaged Kokkos host views from existing arrays on the host

    if (nbat->kk_nbat == NULL)
    {
        snew(nbat->kk_nbat,1);    
    }

    nbat->kk_nbat->h_un_Ftab = HAT::t_un_real_1d(ic->tabq_coul_F, ic->tabq_size);
    nbat->kk_nbat->h_un_shiftvec = HAT::t_un_real_1d(shift_vec[0], SHIFTS);

    // initialize Kokkos functor

    const int teamsize = 1;
    const int nteams = nnbl;//int(nnbl/teamsize);
    const int nvectors = 8;

    // time_t start,end;
    // time (&start);
    typedef nbnxn_kokkos_kernel_functor f_type;
    f_type nb_f(nbat->kk_nbat->h_un_Ftab,
                HAT::t_un_real_1d(ic->tabq_coul_V, ic->tabq_size),
                nbat->kk_nbat->h_un_shiftvec,
                ic->rcoulomb*ic->rcoulomb,
                ic->epsfac,
                ic->tabq_scale,
                ic->repulsion_shift.cpot,
                ic->dispersion_shift.cpot,
                ic->sh_ewald,
                nbat->ntype,
                nbat,
                nbl,
                nnbl);
    // time (&end);
    // double dif = difftime (end,start);
    // printf ("Constructor elasped time is %.2lf seconds.", dif );

    // Kokkos::DefaultExecutionSpace::print_configuration(std::cout,true);
    //Kokkos::TeamPolicy<typename f_type::device_type> config(nteams,teamsize,nvectors);

    // time (&start);
    // launch kernel

    // if (nnbl == 1)
    //{
    //    Kokkos::parallel_for(config,nb_f);
    //}
    // else
    // {
    //         Kokkos::parallel_for(Kokkos::RangePolicy<typename f_type::device_type,1>(nnbl),nb_f);
    Kokkos::parallel_for(Kokkos::RangePolicy<typename f_type::device_type>(0,nnbl),nb_f);
    // }

    // for now, wait until parallel_for finishes
    Kokkos::fence();


    reduce_energies_over_lists(nbat, nnbl, Vvdw, Vc);

    // time (&end);
    // dif = difftime (end,start);
    // printf ("Paralle_for elasped time is %.2lf seconds.", dif );

    // // print forces
    // printf("Total forces\n");
    // int i = 1000;
    // // for ( i = 0; i < nbat->nalloc; i++)
    // {
    //     printf("i = %d, fx = %lf\n", i, out->f[i*FI_STRIDE+XX]);
    //     printf("i = %d, fy = %lf\n", i, out->f[i*FI_STRIDE+YY]);
    //     printf("i = %d, fz = %lf\n", i, out->f[i*FI_STRIDE+ZZ]);
    // }

}

#undef UNROLLI
#undef UNROLLJ
#undef XI_STRIDE
#undef FI_STRIDE
#undef NBNXN_KOKKOS_CLUSTER_I_SIZE
#undef NBNXN_KOKKOS_CLUSTER_J_SIZE


//         int n;
//         int cjind0, cjind1;
//         int ci, ci_sh;
//         int ish, ishf;
//         int cj;

//         const int I = dev.league_rank() * dev.team_size() + dev.team_rank();

//         size_t x_size  = NBNXN_KOKKOS_CLUSTER_I_SIZE*XI_STRIDE*sizeof(real);
//         real* xi_shmem = (real * ) dev.team_shmem().get_shmem(x_size);
//         real* xj_shmem = (real * ) dev.team_shmem().get_shmem(x_size);

//         size_t q_size  = NBNXN_KOKKOS_CLUSTER_I_SIZE*sizeof(real);
//         real* qi_shmem = (real * ) dev.team_shmem().get_shmem(q_size);
//         real* qj_shmem = (real * ) dev.team_shmem().get_shmem(q_size);

//         size_t f_size  = NBNXN_KOKKOS_CLUSTER_I_SIZE*FI_STRIDE*sizeof(real);
//         real* fi_shmem = (real * ) dev.team_shmem().get_shmem(f_size);

//         struct fj_reduce
//         {
//             real f[NBNXN_KOKKOS_CLUSTER_J_SIZE*FI_STRIDE];

//             KOKKOS_INLINE_FUNCTION
//             void operator+=(const volatile struct fj_reduce &rhs) volatile
//             {

//                 for (int j = 0; j < NBNXN_KOKKOS_CLUSTER_J_SIZE*FI_STRIDE; j++)
//                 {
//                     f[j] += rhs.f[j];
//                 }
//             }

//         };

//         struct fj_reduce fj_sh;

//         for (n = 0; n < nci_(I); n++)
//         {
//             // \todo here NBNXN_CI_SHIFT=127, find out is this the right multiplier for kokkos kernel
//             ish              = (ci_[I](n).shift & NBNXN_CI_SHIFT);
//             /* x, f and fshift are assumed to be stored with stride 3 */
//             ishf             = ish*DIM;
//             cjind0           = ci_[I](n).cj_ind_start;
//             cjind1           = ci_[I](n).cj_ind_end;
//             /* Currently only works super-cells equal to sub-cells */
//             ci               = ci_[I](n).ci;
//             ci_sh            = (ish == CENTRAL ? ci : -1);

//             // load cluster i coordinates into shared memory
//             Kokkos::parallel_for
//                 (Kokkos::ThreadVectorRange(dev,NBNXN_KOKKOS_CLUSTER_I_SIZE), [&] (const int& k)
//                  {
//                      int ai = ci * NBNXN_KOKKOS_CLUSTER_I_SIZE + k;
//                      xi_shmem[k*XI_STRIDE + XX] = x_(ai*XI_STRIDE + XX) + shiftvec_(ishf + XX);
//                      xi_shmem[k*XI_STRIDE + YY] = x_(ai*XI_STRIDE + YY) + shiftvec_(ishf + YY);
//                      xi_shmem[k*XI_STRIDE + ZZ] = x_(ai*XI_STRIDE + ZZ) + shiftvec_(ishf + ZZ);
//                      qi_shmem[k] = facel_ * q_(ai);
//                      fi_shmem[k*FI_STRIDE + XX] = 0.0;
//                      fi_shmem[k*FI_STRIDE + YY] = 0.0;
//                      fi_shmem[k*FI_STRIDE + ZZ] = 0.0;
//                  });

//             int cjind = cjind0;

// #define CHECK_EXCLS

//             while (cjind < cjind1 && cj_[I](cjind).excl != 0xffff)
//             {
// #include "gromacs/mdlib/nbnxn_kokkos/nbnxn_kokkos_kernel_inner.h"
//                 cjind++;
//             } 

// #undef CHECK_EXCLS

//             for (; (cjind < cjind1); cjind++)
//             {
// #include "gromacs/mdlib/nbnxn_kokkos/nbnxn_kokkos_kernel_inner.h"
//             }

                
//             // add i forces to global f
//             // atomic addition
//             Kokkos::parallel_for
//                 (Kokkos::ThreadVectorRange(dev,NBNXN_KOKKOS_CLUSTER_I_SIZE), [&] (const int& k)
//                  {
//                      int ai = ci * NBNXN_KOKKOS_CLUSTER_I_SIZE + k;
//                      f_[I](ai*FI_STRIDE + XX) += fi_shmem[k*FI_STRIDE + XX];
//                      f_[I](ai*FI_STRIDE + YY) += fi_shmem[k*FI_STRIDE + YY];
//                      f_[I](ai*FI_STRIDE + ZZ) += fi_shmem[k*FI_STRIDE + ZZ];
//                  });

//         } // for loop over ci



// // following loop structure based on clusters explained in fig. 6
// // of ref. S. Pall and B. Hess, Comp. Phys. Comm., 184, 2013
// // for each cj cluster
// //   load N coords+params for cj
// //   for j = 0 to M
// //     for i = 0 to N
// //       calculate interaction of ci*M+i  with cj*N+j
// //   store N cj-forces
// // store M ci-forces
