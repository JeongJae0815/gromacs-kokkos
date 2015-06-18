/*
 * This file is part of the GROMACS molecular simulation package.
 *
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
/*! \file
 *  \brief Define 
 *
 *  \author Sikandar Y. Mashayak <symashayak@gmail.com>
 */

#include "gmxpre.h"



#include <assert.h>
#include <math.h>
#include <string.h>

#include "gromacs/gmxlib/kokkos_tools/kokkos_macros.h"
#include "gromacs/gmxlib/kokkos_tools/kokkos_memory.h"
#include "gromacs/gmxlib/kokkos_tools/kokkos_type.h"
#include "gromacs/legacyheaders/gmx_omp_nthreads.h"
#include "gromacs/legacyheaders/macros.h"
#include "gromacs/legacyheaders/nrnb.h"
#include "gromacs/legacyheaders/ns.h"
#include "gromacs/legacyheaders/types/commrec.h"
#include "gromacs/math/utilities.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdlib/nb_verlet.h"
#include "gromacs/mdlib/nbnxn_atomdata.h"
#include "gromacs/mdlib/nbnxn_consts.h"
#include "gromacs/mdlib/nbnxn_internal.h"
#include "gromacs/mdlib/nbnxn_search.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/simd/simd.h"
#include "gromacs/simd/vector_operations.h"
#include "gromacs/utility/smalloc.h"

#include "nbnxn_kokkos_types.h"

#ifdef NBNXN_SEARCH_BB_SIMD4
/* Always use 4-wide SIMD for bounding box calculations */

#    ifndef GMX_DOUBLE
/* Single precision BBs + coordinates, we can also load coordinates with SIMD */
#        define NBNXN_SEARCH_SIMD4_FLOAT_X_BB
#    endif

#    if defined NBNXN_SEARCH_SIMD4_FLOAT_X_BB && (GPU_NSUBCELL == 4 || GPU_NSUBCELL == 8)
/* Store bounding boxes with x, y and z coordinates in packs of 4 */
#        define NBNXN_PBB_SIMD4
#    endif

/* The packed bounding box coordinate stride is always set to 4.
 * With AVX we could use 8, but that turns out not to be faster.
 */
#    define STRIDE_PBB        4
#    define STRIDE_PBB_2LOG   2

#endif /* NBNXN_SEARCH_BB_SIMD4 */

#ifdef GMX_NBNXN_SIMD

/* The functions below are macros as they are performance sensitive */

/* 4x4 list, pack=4: no complex conversion required */
/* i-cluster to j-cluster conversion */
#define CI_TO_CJ_J4(ci)   (ci)
/* cluster index to coordinate array index conversion */
#define X_IND_CI_J4(ci)  ((ci)*STRIDE_P4)
#define X_IND_CJ_J4(cj)  ((cj)*STRIDE_P4)

/* 4x2 list, pack=4: j-cluster size is half the packing width */
/* i-cluster to j-cluster conversion */
#define CI_TO_CJ_J2(ci)  ((ci)<<1)
/* cluster index to coordinate array index conversion */
#define X_IND_CI_J2(ci)  ((ci)*STRIDE_P4)
#define X_IND_CJ_J2(cj)  (((cj)>>1)*STRIDE_P4 + ((cj) & 1)*(PACK_X4>>1))

/* 4x8 list, pack=8: i-cluster size is half the packing width */
/* i-cluster to j-cluster conversion */
#define CI_TO_CJ_J8(ci)  ((ci)>>1)
/* cluster index to coordinate array index conversion */
#define X_IND_CI_J8(ci)  (((ci)>>1)*STRIDE_P8 + ((ci) & 1)*(PACK_X8>>1))
#define X_IND_CJ_J8(cj)  ((cj)*STRIDE_P8)

/* The j-cluster size is matched to the SIMD width */
#if GMX_SIMD_REAL_WIDTH == 2
#define CI_TO_CJ_SIMD_4XN(ci)  CI_TO_CJ_J2(ci)
#define X_IND_CI_SIMD_4XN(ci)  X_IND_CI_J2(ci)
#define X_IND_CJ_SIMD_4XN(cj)  X_IND_CJ_J2(cj)
#else
#if GMX_SIMD_REAL_WIDTH == 4
#define CI_TO_CJ_SIMD_4XN(ci)  CI_TO_CJ_J4(ci)
#define X_IND_CI_SIMD_4XN(ci)  X_IND_CI_J4(ci)
#define X_IND_CJ_SIMD_4XN(cj)  X_IND_CJ_J4(cj)
#else
#if GMX_SIMD_REAL_WIDTH == 8
#define CI_TO_CJ_SIMD_4XN(ci)  CI_TO_CJ_J8(ci)
#define X_IND_CI_SIMD_4XN(ci)  X_IND_CI_J8(ci)
#define X_IND_CJ_SIMD_4XN(cj)  X_IND_CJ_J8(cj)
/* Half SIMD with j-cluster size */
#define CI_TO_CJ_SIMD_2XNN(ci) CI_TO_CJ_J4(ci)
#define X_IND_CI_SIMD_2XNN(ci) X_IND_CI_J4(ci)
#define X_IND_CJ_SIMD_2XNN(cj) X_IND_CJ_J4(cj)
#else
#if GMX_SIMD_REAL_WIDTH == 16
#define CI_TO_CJ_SIMD_2XNN(ci) CI_TO_CJ_J8(ci)
#define X_IND_CI_SIMD_2XNN(ci) X_IND_CI_J8(ci)
#define X_IND_CJ_SIMD_2XNN(cj) X_IND_CJ_J8(cj)
#else
#error "unsupported GMX_SIMD_REAL_WIDTH"
#endif
#endif
#endif
#endif

#endif /* GMX_NBNXN_SIMD */


#ifdef NBNXN_SEARCH_BB_SIMD4
/* Store bounding boxes corners as quadruplets: xxxxyyyyzzzz */
#define NBNXN_BBXXXX
/* Size of bounding box corners quadruplet */
#define NNBSBB_XXXX      (NNBSBB_D*DIM*STRIDE_PBB)
#endif

/* We shift the i-particles backward for PBC.
 * This leads to more conditionals than shifting forward.
 * We do this to get more balanced pair lists.
 */
#define NBNXN_SHIFT_BACKWARD


/* This define is a lazy way to avoid interdependence of the grid
 * and searching data structures.
 */
#define NBNXN_NA_SC_MAX (GPU_NSUBCELL*NBNXN_GPU_CLUSTER_SIZE)

/* Initializes a single nbnxn_pairlist_t data structure */
static void nbnxn_init_pairlist_fep(t_nblist *nl)
{
    nl->type        = GMX_NBLIST_INTERACTION_FREE_ENERGY;
    nl->igeometry   = GMX_NBLIST_GEOMETRY_PARTICLE_PARTICLE;
    /* The interaction functions are set in the free energy kernel fuction */
    nl->ivdw        = -1;
    nl->ivdwmod     = -1;
    nl->ielec       = -1;
    nl->ielecmod    = -1;

    nl->maxnri      = 0;
    nl->maxnrj      = 0;
    nl->nri         = 0;
    nl->nrj         = 0;
    nl->iinr        = NULL;
    nl->gid         = NULL;
    nl->shift       = NULL;
    nl->jindex      = NULL;
    nl->jjnr        = NULL;
    nl->excl_fep    = NULL;

}

/* Initializes a single nbnxn_pairlist_t data structure */
static void nbnxn_init_pairlist_kokkos(nbnxn_pairlist_t *nbl,
                                       gmx_bool          bSimple,
                                       nbnxn_alloc_t    *alloc,
                                       nbnxn_free_t     *free)
{
    if (alloc == NULL)
    {
        nbl->alloc = nbnxn_alloc_aligned;
    }
    else
    {
        nbl->alloc = alloc;
    }
    if (free == NULL)
    {
        nbl->free = nbnxn_free_aligned;
    }
    else
    {
        nbl->free = free;
    }

    nbl->bSimple     = bSimple;
    nbl->na_sc       = 0;
    nbl->na_ci       = 0;
    nbl->na_cj       = 0;
    nbl->nci         = 0;

    /* Set Kokkos dual View and host pointer of ci to NULL */
    //    nbl->ci          = NULL;
    destroy_kokkos(nbl->kk_plist->k_ci,nbl->ci);

    nbl->ci_nalloc   = 0;
    nbl->ncj         = 0;

    /* Set Kokkos dual View and host pointer of ci to NULL */
    //    nbl->cj          = NULL;
    destroy_kokkos(nbl->kk_plist->k_cj,nbl->cj);

    nbl->cj_nalloc   = 0;
    nbl->ncj4        = 0;
    /* We need one element extra in sj, so alloc initially with 1 */
    nbl->cj4_nalloc  = 0;

    /* Set Kokkos dual View and host pointer of ci to NULL */
    nbl->cj4          = NULL;
    //destroy_kokkos(nbl->kk_plist->k_cj4,nbl->cj4);

    nbl->nci_tot     = 0;

    // if (!nbl->bSimple)
    // {
    //     nbl->excl        = NULL;
    //     nbl->excl_nalloc = 0;
    //     nbl->nexcl       = 0;
    //     check_excl_space(nbl, 1);
    //     nbl->nexcl       = 1;
    //     set_no_excls(&nbl->excl[0]);
    // }

    snew(nbl->work, 1);
    if (nbl->bSimple)
    {
        snew_aligned(nbl->work->bb_ci, 1, NBNXN_SEARCH_BB_MEM_ALIGN);
    }
    else
    {
#ifdef NBNXN_BBXXXX
        snew_aligned(nbl->work->pbb_ci, GPU_NSUBCELL/STRIDE_PBB*NNBSBB_XXXX, NBNXN_SEARCH_BB_MEM_ALIGN);
#else
        snew_aligned(nbl->work->bb_ci, GPU_NSUBCELL, NBNXN_SEARCH_BB_MEM_ALIGN);
#endif
    }
    snew_aligned(nbl->work->x_ci, NBNXN_NA_SC_MAX*DIM, NBNXN_SEARCH_BB_MEM_ALIGN);
#ifdef GMX_NBNXN_SIMD
    snew_aligned(nbl->work->x_ci_simd_4xn, 1, NBNXN_MEM_ALIGN);
    snew_aligned(nbl->work->x_ci_simd_2xnn, 1, NBNXN_MEM_ALIGN);
#endif
    snew_aligned(nbl->work->d2, GPU_NSUBCELL, NBNXN_SEARCH_BB_MEM_ALIGN);

    nbl->work->sort            = NULL;
    nbl->work->sort_nalloc     = 0;
    nbl->work->sci_sort        = NULL;
    nbl->work->sci_sort_nalloc = 0;
}

/* Initializes a set of pair lists with Kokkos views */
void nbnxn_init_pairlist_set_kokkos(nbnxn_pairlist_set_t *nbl_list,
                                    gmx_bool bSimple, gmx_bool bCombined,
                                    nbnxn_alloc_t *alloc,
                                    nbnxn_free_t  *free)
{
    int i;

    nbl_list->bSimple   = bSimple;
    nbl_list->bCombined = bCombined;

    nbl_list->nnbl = gmx_omp_nthreads_get(emntNonbonded);

    if (!nbl_list->bCombined &&
        nbl_list->nnbl > NBNXN_BUFFERFLAG_MAX_THREADS)
    {
        gmx_fatal(FARGS, "%d OpenMP threads were requested. Since the non-bonded force buffer reduction is prohibitively slow with more than %d threads, we do not allow this. Use %d or less OpenMP threads.",
                  nbl_list->nnbl, NBNXN_BUFFERFLAG_MAX_THREADS, NBNXN_BUFFERFLAG_MAX_THREADS);
    }

    snew(nbl_list->nbl, nbl_list->nnbl);
    snew(nbl_list->nbl_fep, nbl_list->nnbl);
    /* Execute in order to avoid memory interleaving between threads */
#pragma omp parallel for num_threads(nbl_list->nnbl) schedule(static)
    for (i = 0; i < nbl_list->nnbl; i++)
    {
        /* Allocate the nblist data structure locally on each thread
         * to optimize memory access for NUMA architectures.
         */
        snew(nbl_list->nbl[i], 1);

        /* Only list 0 is used on the GPU, use normal allocation for i>0 */
        if (i == 0)
        {
            nbnxn_init_pairlist_kokkos(nbl_list->nbl[i], nbl_list->bSimple, alloc, free);
        }
        else
        {
            nbnxn_init_pairlist_kokkos(nbl_list->nbl[i], nbl_list->bSimple, NULL, NULL);
        }

        snew(nbl_list->nbl_fep[i], 1);
        nbnxn_init_pairlist_fep(nbl_list->nbl_fep[i]);
    }
}

/* Deallocates a set of pair lists with Kokkos views */
void nbnxn_destroy_pairlist_set_kokkos(nbnxn_pairlist_set_t *nbl_list,
                                       gmx_bool bSimple, gmx_bool bCombined)
{

}

/* Allocate new ci entry with Kokkos views */
void new_ci_entry_kokkos(nbnxn_pairlist_t *nbl, int ci, int shift, int flags)
{
    if (nbl->nci + 1 > nbl->ci_nalloc)
    {
        //        nb_realloc_ci(nbl, nbl->nci+1);
        nbl->ci_nalloc = over_alloc_small(nbl->nci+1);
        grow_kokkos(nbl->kk_plist->k_ci,nbl->ci,nbl->ci_nalloc,"plist::ci");
    }

    nbl->ci[nbl->nci].ci            = ci;
    nbl->ci[nbl->nci].shift         = shift;
    /* Store the interaction flags along with the shift */
    nbl->ci[nbl->nci].shift        |= flags;
    nbl->ci[nbl->nci].cj_ind_start  = nbl->ncj;
    nbl->ci[nbl->nci].cj_ind_end    = nbl->ncj;
}

/* Ensures there is enough space for ncell extra j-cells in the list with Kokkos views*/
void check_subcell_list_space_simple_kokkos(nbnxn_pairlist_t *nbl,
                                            int               ncell)
{

    int cj_max;

    cj_max = nbl->ncj + ncell;

    if (cj_max > nbl->cj_nalloc)
    {
        nbl->cj_nalloc = over_alloc_small(cj_max);
        grow_kokkos(nbl->kk_plist->k_cj,nbl->cj,nbl->cj_nalloc,"plist::cj");
    }

}

void nbnxn_sync_pairlist_kokkos(nbnxn_pairlist_t *nbl)
{

    nbl->kk_plist->k_ci.modify<GMXHostType>();
    nbl->kk_plist->k_cj.modify<GMXHostType>();

    nbl->kk_plist->k_ci.sync<GMXDeviceType>();
    nbl->kk_plist->k_cj.sync<GMXDeviceType>();
}
/* Kokkos: copy pairlist from host to device views*/
void nbnxn_kokkos_sync_pairlist(nbnxn_pairlist_set_t *nbl_list)
{

    int i;

    for (i = 0; i < nbl_list->nnbl; i++)
    {
        nbnxn_sync_pairlist_kokkos(nbl_list->nbl[i]);
    }
}
