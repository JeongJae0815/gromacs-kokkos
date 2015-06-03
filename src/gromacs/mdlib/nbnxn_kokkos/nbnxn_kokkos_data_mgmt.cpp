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

#include "config.h"

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "gromacs/mdlib/nbnxn_kokkos_data_mgmt.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/smalloc.h"

#include "nbnxn_kokkos_types.h"

void nbnxn_kokkos_init(FILE                 *fplog,
		       gmx_nbnxn_kokkos_t   **p_nb)
{

    gmx_nbnxn_kokkos_t *nb;

    snew(nb, 1);
    snew(nb->atdat, 1);

    *p_nb = nb;

    printf("\n \n Initialized Kokkos data structures.  \n \n");

    if (debug)
    {
        fprintf(debug, "Initialized Kokkos data structures.\n");
    }

}

void nbnxn_kokkos_finalize()
{

}

void nbnxn_kokkos_init_atomdata(gmx_nbnxn_kokkos_t       *nb,
				const struct nbnxn_atomdata_t *nbat)
{
    int            nalloc, natoms;
    bool           realloced;
    kokkos_atomdata_t *d_atdat   = nb->atdat;

    natoms    = nbat->natoms;
    realloced = false;

    /* need to reallocate if we have to copy more atoms than the amount of space
       available and only allocate if we haven't initialized yet, i.e d_atdat->natoms == -1 */
    if (natoms > d_atdat->nalloc)
    {
        nalloc = over_alloc_small(natoms);

        /* free up first if the arrays have already been initialized */
        if (d_atdat->nalloc != -1)
        {
            // cu_free_buffered(d_atdat->f, &d_atdat->natoms, &d_atdat->nalloc);
            // cu_free_buffered(d_atdat->xq);
            // cu_free_buffered(d_atdat->atom_types);
        }

        // stat = cudaMalloc((void **)&d_atdat->f, nalloc*sizeof(*d_atdat->f));
        // CU_RET_ERR(stat, "cudaMalloc failed on d_atdat->f");
        // stat = cudaMalloc((void **)&d_atdat->xq, nalloc*sizeof(*d_atdat->xq));
        // CU_RET_ERR(stat, "cudaMalloc failed on d_atdat->xq");

        // stat = cudaMalloc((void **)&d_atdat->atom_types, nalloc*sizeof(*d_atdat->atom_types));
        // CU_RET_ERR(stat, "cudaMalloc failed on d_atdat->atom_types");

        d_atdat->nalloc = nalloc;
        realloced       = true;
    }

    d_atdat->natoms       = natoms;
    d_atdat->natoms_local = nbat->natoms_local;

    /* need to clear GPU f output if realloc happened */
    if (realloced)
    {
        // nbnxn_cuda_clear_f(nb, nalloc);
    }

    // copy data from host to device
    // cu_copy_H2D_async(d_atdat->atom_types, nbat->type,
    //                   natoms*sizeof(*d_atdat->atom_types), ls);

}
