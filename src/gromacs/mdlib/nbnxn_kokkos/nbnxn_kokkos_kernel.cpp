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

#include "gromacs/gmxlib/kokkos_tools/kokkos_type.h"
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
#include "gromacs/timing/walltime_accounting.h"

/* We could use nbat->xstride and nbat->fstride, but macros might be faster */
#define XI_STRIDE   3
#define FI_STRIDE   3

#define XJ_STRIDE   3
#define FJ_STRIDE   3

/* For Kokkos, for now, using cluster sizes same as CPU i.e. 4 */
#define NBNXN_KOKKOS_CLUSTER_I_SIZE NBNXN_CPU_CLUSTER_I_SIZE // 4
#define NBNXN_KOKKOS_CLUSTER_J_SIZE NBNXN_CPU_CLUSTER_I_SIZE // 4

#define UNROLLI NBNXN_KOKKOS_CLUSTER_I_SIZE
#define UNROLLJ NBNXN_KOKKOS_CLUSTER_J_SIZE

struct nbnxn_kokkos_kernel_functor
{
    // for now focusing on Xeon Phi device which is run in a native mode, i.e., host==device
    typedef GMXDeviceType device_type;
    typedef GMXDeviceType::execution_space::scratch_memory_space shared_space;

    typedef Kokkos::View<real[UNROLLI][3],shared_space,Kokkos::MemoryUnmanaged> shared_xi;
    typedef Kokkos::View<real[UNROLLI][3],shared_space,Kokkos::MemoryUnmanaged> shared_fi;
    typedef Kokkos::View<real[UNROLLI],shared_space,Kokkos::MemoryUnmanaged> shared_qi;

    typedef Kokkos::View<real[UNROLLJ][3],shared_space,Kokkos::MemoryUnmanaged> shared_xj;
    typedef Kokkos::View<real[UNROLLJ][3],shared_space,Kokkos::MemoryUnmanaged> shared_fj;
    typedef Kokkos::View<real[UNROLLJ],shared_space,Kokkos::MemoryUnmanaged> shared_qj;

    // list of structures needed for non-bonded interactions
    DAT::t_un_real_1d3 x_;
    DAT::t_un_real_1d q_;
    DAT::t_un_int_1d type_;
    DAT::t_un_real_1d nbfp_;
    DAT::t_un_ci_1d* ci_;
    DAT::t_un_cj_1d* cj_;
    // make this atomic when vectorizing because more than one thread may write into same location
    DAT::t_un_real_1d3* f_;
    DAT::t_un_real_1d* Vvdw_;
    DAT::t_un_real_1d* Vc_;
    DAT::t_un_real_1d Ftab_;
    DAT::t_un_real_1d Vtab_;
    DAT::t_un_real_1d shiftvec_;

    typedef Kokkos::
    View<int*, Kokkos::LayoutRight, GMXHostType> t_int_1d;
    t_int_1d nci_;

    const real rcut2_, facel_, tabq_scale_;
    const real repulsion_shift_cpot_, dispersion_shift_cpot_;
    const real sh_ewald_;
    const int ntype_;
    const int nnbl_;

    const int i_vec_[16] = {0,0,0,0,
                            1,1,1,1,
                            2,2,2,2,
                            3,3,3,3};
    const int j_vec_[16] = {0,1,2,3,
                            0,1,2,3,
                            0,1,2,3};

    const real inv_12_ = 1.0/12.0;
    const real inv_6_ = 1.0/6.0;

    struct f_reduce
    {
        real fi[UNROLLI*FI_STRIDE];
        real fj[UNROLLJ*FJ_STRIDE];
        real Vvdw_ci, Vc_ci;

        f_reduce() {
            Vvdw_ci = 0.0;
            Vc_ci   = 0.0;
            for (int i = 0; i < UNROLLI*FI_STRIDE; i++)
            {
                fi[i] =  0.0;
            }
            for (int j = 0; j < UNROLLJ*FJ_STRIDE; j++)
            {
                fj[j] = 0.0;
            }
        }

        KOKKOS_INLINE_FUNCTION
        void operator+=(const volatile struct f_reduce &rhs) volatile
        {
            Vvdw_ci += rhs.Vvdw_ci;
            Vc_ci   += rhs.Vc_ci;

            for (int i = 0; i < UNROLLI*FI_STRIDE; i++)
            {
                fi[i] += rhs.fi[i];
            }

            for (int j = 0; j < UNROLLJ*FJ_STRIDE; j++)
            {
                fj[j] += rhs.fj[j];
            }
        }
    };

    nbnxn_kokkos_kernel_functor (const DAT::t_un_real_1d &Ftab,
                                 const DAT::t_un_real_1d &Vtab,
                                 const DAT::t_un_real_1d &shiftvec,
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

        x_    = DAT::t_un_real_1d3(nbat->x, nbat->nalloc);// * nbat->xstride);
        q_    = DAT::t_un_real_1d(nbat->q, nbat->nalloc);
        type_ = DAT::t_un_int_1d(nbat->type, nbat->ntype);
        nbfp_ = DAT::t_un_real_1d(nbat->nbfp, nbat->ntype*nbat->ntype*2);

        snew(ci_,nnbl_);
        snew(cj_,nnbl_);
        nci_ = t_int_1d("nci",nnbl_);

        snew(f_,nnbl_);
        snew(Vvdw_,nnbl_);
        snew(Vc_,nnbl_);

        for (nb = 0; nb < nnbl_; nb++)
        {
            ci_[nb]   = DAT::t_un_ci_1d(nbl[nb]->ci,nbl[nb]->ci_nalloc);
            cj_[nb]   = DAT::t_un_cj_1d(nbl[nb]->cj,nbl[nb]->cj_nalloc);
            nci_(nb)  = nbl[nb]->nci;
            f_[nb]    = DAT::t_un_real_1d3(nbat->out[nb].f, nbat->nalloc);// * nbat->fstride);
            Vvdw_[nb] = DAT::t_un_real_1d(nbat->out[nb].Vvdw, nbat->out[nb].nV);
            Vc_[nb]   = DAT::t_un_real_1d(nbat->out[nb].Vc, nbat->out[nb].nV);
        }

    };

    ~nbnxn_kokkos_kernel_functor () {};

    KOKKOS_INLINE_FUNCTION
    void operator()(const typename Kokkos::TeamPolicy<device_type>::member_type& dev) const
    {
        const int I = dev.league_rank() * dev.team_size() + dev.team_rank();

        real                facel;
        int                 n, ci, ci_sh;
        int                 ish, ishf;
        gmx_bool            do_LJ, half_LJ, do_coul, do_self;
        int                 cjind0, cjind1, cjind;
        int                 ip, jp;
        real                Vvdw_ci, Vc_ci;
        const real          tabscale = tabq_scale_;
        const real          halfsp = 0.5/tabscale;
        const real          rcut2 = rcut2_;
        const int           ntype2 = ntype_*2;
        int ninner;

        shared_xi xi(dev.team_shmem());
        shared_fi fi(dev.team_shmem());
        shared_qi qi(dev.team_shmem());

        shared_xj xj(dev.team_shmem());
        shared_fj fj(dev.team_shmem());
        shared_qj qj(dev.team_shmem());

        Kokkos::single(Kokkos::PerTeam(dev),[=] () {

                Vvdw_[I](0) = 0.0;
                Vc_[I](0) = 0.0;

            });

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

            // load cluster i coordinates into shared memory
            Kokkos::parallel_for
                (Kokkos::ThreadVectorRange(dev,UNROLLI), [&] (const int& k)
                 {
                     int aik = ci * UNROLLI + k;

                     xi(k,XX) = x_(aik,XX) + shiftvec_(ishf + XX);
                     xi(k,YY) = x_(aik,YY) + shiftvec_(ishf + YY);
                     xi(k,ZZ) = x_(aik,ZZ) + shiftvec_(ishf + ZZ);

                     qi(k) =  q_(aik);

                     fi(k,XX) = 0.0;
                     fi(k,YY) = 0.0;
                     fi(k,ZZ) = 0.0;
            
                 });

            if (do_self)
            {
                real Vc_sub_self = 0.5*Vtab_(0);

                if (cj_[I](cjind0).cj == ci_sh)
                {
                    real Vc_sum = 0.0;
                    Kokkos::parallel_reduce
                        (Kokkos::ThreadVectorRange(dev,UNROLLI), [=] (int& k, real& Vc)
                         {
                             Vc += facel_ * qi(k) * qi(k) * Vc_sub_self;
                         }, Vc_sum);

                    Kokkos::single(Kokkos::PerTeam(dev),[=] () {
                            Vc_[I](0) -= Vc_sum;
                        });                    
                }
            }

            Kokkos::parallel_for
                (Kokkos::ThreadVectorRange(dev,UNROLLI), [&] (const int& k)
                 {
                     qi(k) *= facel_;
                 });

            cjind = cjind0;

            while(cjind < cjind1 && cj_[I](cjind).excl != 0xffff)
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

            Kokkos::parallel_for
                (Kokkos::ThreadVectorRange(dev,UNROLLI), [&] (const int& k)
                 {
                     int aik = ci * UNROLLI + k;

                     f_[I](aik, XX) += fi(k,XX);
                     f_[I](aik, YY) += fi(k,YY);
                     f_[I](aik, ZZ) += fi(k,ZZ);
                 });

            Kokkos::single(Kokkos::PerTeam(dev),[=] () {

                    Vvdw_[I](0) += Vvdw_ci;
                    Vc_[I](0) += Vc_ci;

                });
            
        }
    } // operator()

    size_t team_shmem_size (int team_size) const {
        return sizeof(real ) * (UNROLLI * XI_STRIDE + UNROLLJ * XJ_STRIDE + // xi + xj size
                                UNROLLI * FI_STRIDE + UNROLLJ * FJ_STRIDE + // fi + fj size
                                UNROLLI + UNROLLJ // qi + qj size
                                );
    }

};

void nbnxn_kokkos_launch_kernel(nbnxn_pairlist_set_t      *nbl_list,
                                nbnxn_atomdata_t          *nbat,
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

    // initialize Kokkos functor
    typedef nbnxn_kokkos_kernel_functor f_type;
    f_type nb_f(DAT::t_un_real_1d(ic->tabq_coul_F, ic->tabq_size),
                DAT::t_un_real_1d(ic->tabq_coul_V, ic->tabq_size),
                DAT::t_un_real_1d(shift_vec[0], SHIFTS),
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

    // typedef Kokkos::RangePolicy<typename f_type::device_type,
    //     Kokkos::Impl::integral_constant<unsigned,1>> range_policy;
    //    double start = gmx_gettime();
    // Kokkos::parallel_for(range_policy(0,nnbl),nb_f);
    //    double end = gmx_gettime();
    //    double elapsed_time = end - start;
    //    printf("Kokkos parallel_for elapsed time %lf \n ",elapsed_time);


    const int teamsize = 1;
    const int nteams = nnbl;
    const int nvectors = 16;

    typedef Kokkos::TeamPolicy<typename f_type::device_type> team_policy;

    team_policy config(nteams,teamsize,nvectors);

    double start = gmx_gettime();
    Kokkos::parallel_for(config,nb_f);
    double end = gmx_gettime();
    double elapsed_time = end - start;
    printf("Kokkos parallel_for elapsed time %lf \n ",elapsed_time);

    Kokkos::fence();

    reduce_energies_over_lists(nbat, nnbl, Vvdw, Vc);

}

#undef UNROLLI
#undef UNROLLJ
#undef XI_STRIDE
#undef FI_STRIDE
#undef NBNXN_KOKKOS_CLUSTER_I_SIZE
#undef NBNXN_KOKKOS_CLUSTER_J_SIZE
