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
    // atomic because more than one thread may write into same location
    HAT::t_un_at_real_1d* f_;
    HAT::t_un_real_1d Ftab_;
    HAT::t_un_real_1d shiftvec_;

    typedef Kokkos::
    View<int*, Kokkos::LayoutRight, GMXHostType> t_int_1d;
    t_int_1d nci_;

    const real rcut2_, facel_, tabq_scale_;
    const int ntype_;
    const int nnbl_;

    nbnxn_kokkos_kernel_functor (const HAT::t_un_real_1d &Ftab,
                                 const HAT::t_un_real_1d &shiftvec,
                                 const real &rcut2,
                                 const real &facel,
                                 const real &tabq_scale,
                                 const int &ntype,
                                 nbnxn_atomdata_t    *nbat,                                 
                                 nbnxn_pairlist_t **nbl,
                                 const int &nnbl):
        Ftab_(Ftab),shiftvec_(shiftvec),
        rcut2_(rcut2),facel_(facel),tabq_scale_(tabq_scale),ntype_(ntype),nnbl_(nnbl)
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

        for (nb = 0; nb < nnbl_; nb++)
        {
            ci_[nb] = HAT::t_un_ci_1d(nbl[nb]->ci,nbl[nb]->ci_nalloc);
            cj_[nb] = HAT::t_un_cj_1d(nbl[nb]->cj,nbl[nb]->cj_nalloc);
            nci_(nb) = nbl[nb]->nci;
            f_[nb] = HAT::t_un_real_1d(nbat->out[nb].f, nbat->nalloc * nbat->fstride);
        }
        
    };

    ~nbnxn_kokkos_kernel_functor ( )
    {

    };

    // with thread teams
    KOKKOS_INLINE_FUNCTION
    void operator()(const  typename Kokkos::TeamPolicy<device_type>::member_type& dev) const
    {
        int n;
        int cjind0, cjind1;
        int ci, ci_sh;
        int ish, ishf;
        int cj;

        const int I = dev.league_rank() * dev.team_size() + dev.team_rank();

        size_t x_size  = NBNXN_KOKKOS_CLUSTER_I_SIZE*X_STRIDE*sizeof(real);
        real* xi_shmem = (real * ) dev.team_shmem().get_shmem(x_size);
        real* xj_shmem = (real * ) dev.team_shmem().get_shmem(x_size);

        size_t q_size  = NBNXN_KOKKOS_CLUSTER_I_SIZE*sizeof(real);
        real* qi_shmem = (real * ) dev.team_shmem().get_shmem(q_size);
        real* qj_shmem = (real * ) dev.team_shmem().get_shmem(q_size);

        size_t f_size  = NBNXN_KOKKOS_CLUSTER_I_SIZE*F_STRIDE*sizeof(real);
        real* fi_shmem = (real * ) dev.team_shmem().get_shmem(f_size);

        struct fj_reduce
        {
            real f[NBNXN_KOKKOS_CLUSTER_J_SIZE*F_STRIDE];

            KOKKOS_INLINE_FUNCTION
            void operator+=(const volatile struct fj_reduce &rhs) volatile
            {

                for (int j = 0; j < NBNXN_KOKKOS_CLUSTER_J_SIZE*F_STRIDE; j++)
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
                         xi_shmem[k*X_STRIDE + XX] = x_(ai*X_STRIDE + XX) + shiftvec_(ishf + XX);
                         xi_shmem[k*X_STRIDE + YY] = x_(ai*X_STRIDE + YY) + shiftvec_(ishf + YY);
                         xi_shmem[k*X_STRIDE + ZZ] = x_(ai*X_STRIDE + ZZ) + shiftvec_(ishf + ZZ);
                         qi_shmem[k] = facel_ * q_(ai);
                         fi_shmem[k*F_STRIDE + XX] = 0.0;
                         fi_shmem[k*F_STRIDE + YY] = 0.0;
                         fi_shmem[k*F_STRIDE + ZZ] = 0.0;
                     });

                int cjind = cjind0;

#define CHECK_EXCLS

                while (cjind < cjind1 && cj_[I](cjind).excl != 0xffff)
                {
                    cj = cj_[I](cjind).cj;

                    //wait until all threads load their xi and xj
                    dev.team_barrier();

                    // load cluster j coordinates into shared memory
                    Kokkos::parallel_for
                        (Kokkos::ThreadVectorRange(dev,NBNXN_KOKKOS_CLUSTER_J_SIZE), [&] (const int& k)
                         {
                             int aj = cj * NBNXN_KOKKOS_CLUSTER_J_SIZE + k;
                             xj_shmem[k*X_STRIDE + XX] = x_(aj*X_STRIDE + XX);
                             xj_shmem[k*X_STRIDE + YY] = x_(aj*X_STRIDE + YY);
                             xj_shmem[k*X_STRIDE + ZZ] = x_(aj*X_STRIDE + ZZ);
                             qj_shmem[k] = facel_ * q_(aj);
                             fj_sh.f[k*F_STRIDE + XX] = 0.0;
                             fj_sh.f[k*F_STRIDE + YY] = 0.0;
                             fj_sh.f[k*F_STRIDE + ZZ] = 0.0;
                         });
                    
                    // compute forces on i atoms in parallel and reduce forces on j
                    Kokkos::parallel_reduce 
                        (Kokkos::ThreadVectorRange(dev,NBNXN_KOKKOS_CLUSTER_I_SIZE), [=] (const int& k, struct fj_reduce& fj)
                         {
                             for (int j = 0; j < NBNXN_KOKKOS_CLUSTER_J_SIZE; j++)
                             {
                                 int ai = ci * NBNXN_KOKKOS_CLUSTER_I_SIZE + k;
                                 int i_type = ntype_*2*type_(ai);
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

                                 interact = ((cj_[I](cjind).excl>>(k*NBNXN_KOKKOS_CLUSTER_I_SIZE + j)) & 1);
#ifndef EXCL_FORCES
                                 skipmask = interact;
#else
                                 skipmask = !(cj == ci_sh && j <= k);
#endif
#else
#define interact 1.0
                                 skipmask = 1.0;
#endif

                                 real dx = xi_shmem[k*X_STRIDE + XX] - xj_shmem[j*X_STRIDE + XX];
                                 real dy = xi_shmem[k*X_STRIDE + YY] - xj_shmem[j*X_STRIDE + YY];
                                 real dz = xi_shmem[k*X_STRIDE + ZZ] - xj_shmem[j*X_STRIDE + ZZ];
                                 real rsq = dx*dx + dy*dy + dz*dz;

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

                                 real rinv = gmx_invsqrt(rsq);
                                 /* 5 flops for invsqrt */

                                 /* Partially enforce the cut-off (and perhaps
                                  * exclusions) to avoid possible overflow of
                                  * rinvsix when computing LJ, and/or overflowing
                                  * the Coulomb table during lookup. */
                                 rinv = rinv * skipmask;

                                 real rinvsq  = rinv*rinv;

                                 // compute Coulomb force for Ewald type
                                 real qq     = skipmask * qi_shmem[k] * qj_shmem[j];
                                 real rs     = rsq*rinv*tabq_scale_;
                                 int  ri     = (int)rs;
                                 real frac   = rs - ri;
                                 real fexcl  = (1 - frac)*Ftab_(ri) + frac*Ftab_(ri+1);

                                 real fcoul  = qq*rinv*(interact*rinvsq - fexcl);

                                 // compute LJ126 simple cut-off force
                                 // \todo using a combination rule may be more memory efficient
                                 int j_global = (cj * NBNXN_KOKKOS_CLUSTER_J_SIZE + j);
                                 int j_type = type_(j_global);

                                 real c6      = nbfp_(i_type + j_type*2 ); //0.26187E-02;   //nbfp_(i_type + j_type*2 );
                                 real c12     = nbfp_(i_type + j_type*2+1 );//0.26307E-05;   //nbfp_(i_type + j_type*2+1 );

                                 // printf("c6 = %lf",c6);
                                 // printf("c12 = %lf",c12);

                                 real rinvsix = interact*rinvsq*rinvsq*rinvsq;
                                 real FrLJ6   = c6*rinvsix;
                                 real FrLJ12  = c12*rinvsix*rinvsix;
                                 real frLJ    = FrLJ12 - FrLJ6;

                                 real fscal = frLJ*rinvsq + fcoul;

                                 real fx = fscal*dx;
                                 real fy = fscal*dy;
                                 real fz = fscal*dz;

                                 /* Increment i-atom force */
                                 fi_shmem[k*F_STRIDE + XX] += fx;
                                 fi_shmem[k*F_STRIDE + YY] += fy;
                                 fi_shmem[k*F_STRIDE + ZZ] += fz;

                                 fj.f[j*F_STRIDE + XX] -= fx;
                                 fj.f[j*F_STRIDE + YY] -= fy;
                                 fj.f[j*F_STRIDE + ZZ] -= fz;

                             } // for loop over j
                            
                         }, fj_sh); // parallel vector loop

                    // add j forces to global f
                    // atomic addition
                    Kokkos::parallel_for
                        (Kokkos::ThreadVectorRange(dev,NBNXN_KOKKOS_CLUSTER_J_SIZE), [&] (const int& k)
                         {
                             int aj = cj * NBNXN_KOKKOS_CLUSTER_I_SIZE + k;
                             f_[I](aj*F_STRIDE + XX) += fj_sh.f[k*F_STRIDE + XX];
                             f_[I](aj*F_STRIDE + YY) += fj_sh.f[k*F_STRIDE + YY];
                             f_[I](aj*F_STRIDE + ZZ) += fj_sh.f[k*F_STRIDE + ZZ];
                         });

                    cjind++;

                } // while loop over cjind

#undef CHECK_EXCLS
                
                // add i forces to global f
                // atomic addition
                Kokkos::parallel_for
                    (Kokkos::ThreadVectorRange(dev,NBNXN_KOKKOS_CLUSTER_I_SIZE), [&] (const int& k)
                     {
                         int ai = ci * NBNXN_KOKKOS_CLUSTER_I_SIZE + k;
                         f_[I](ai*F_STRIDE + XX) += fi_shmem[k*F_STRIDE + XX];
                         f_[I](ai*F_STRIDE + YY) += fi_shmem[k*F_STRIDE + YY];
                         f_[I](ai*F_STRIDE + ZZ) += fi_shmem[k*F_STRIDE + ZZ];
                     });

            } // for loop over ci
     
                 } // operator()
        
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

        typedef nbnxn_kokkos_kernel_functor f_type;
        f_type nb_f(nbat->kk_nbat->h_un_Ftab,
                    nbat->kk_nbat->h_un_shiftvec,
                    ic->rcoulomb*ic->rcoulomb,
                    ic->epsfac,
                    ic->tabq_scale,
                    nbat->ntype,
                    nbat,
                    nbl,
                    nnbl);

        // Kokkos::DefaultExecutionSpace::print_configuration(std::cout,true);
        Kokkos::TeamPolicy<typename f_type::device_type> config(nteams,teamsize,nvectors);

        // launch kernel
        Kokkos::parallel_for(config,nb_f);

        // for now, wait until parallel_for finishes
        Kokkos::fence();

        // // print forces
        // printf("Total forces\n");
        // int i = 1000;
        // // for ( i = 0; i < nbat->nalloc; i++)
        // {
        //     printf("i = %d, fx = %lf\n", i, out->f[i*F_STRIDE+XX]);
        //     printf("i = %d, fy = %lf\n", i, out->f[i*F_STRIDE+YY]);
        //     printf("i = %d, fz = %lf\n", i, out->f[i*F_STRIDE+ZZ]);
        // }

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
