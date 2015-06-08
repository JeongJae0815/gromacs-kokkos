/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2015, by the GROMACS development team, led by Mark
 * Abraham, David van der Spoel, Berk Hess, and Erik Lindahl, and
 * including many others, as listed in the AUTHORS file in the
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
/*! \libinternal \file
 *  \brief Declare interface for Kokkos data transfer for NBNXN module
 *
 *  \author Sikandar Y. Mashayak <symashayak@gmail.com>
 *  \ingroup module_mdlib
 *  \inlibraryapi
 */

#ifndef NBNXN_KOKKOS_DATA_MGMT_H
#define NBNXN_KOKKOS_DATA_MGMT_H

#include "gromacs/legacyheaders/types/interaction_const.h"
#include "gromacs/legacyheaders/types/simple.h"
#include "gromacs/mdlib/nbnxn_kokkos/kokkos_macros.h"
#include "gromacs/mdlib/nbnxn_kokkos_types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct nbnxn_atomdata_t;

/** Initializes the data structures related to Kokkos nonbonded calculations. */
KOKKOS_FUNC_QUALIFIER
void nbnxn_kokkos_init(FILE gmx_unused                        *fplog,
		       gmx_nbnxn_kokkos_t gmx_unused          **p_nb) KOKKOS_FUNC_TERM

/** Initializes simulation constant data. */
KOKKOS_FUNC_QUALIFIER
void nbnxn_kokkos_init_const(gmx_nbnxn_kokkos_t gmx_unused                    *nb,
			     const interaction_const_t      gmx_unused        *ic,
			     const struct nonbonded_verlet_group_t gmx_unused *nbv_group) KOKKOS_FUNC_TERM

/** Deallocates the Kokkos views. */
KOKKOS_FUNC_QUALIFIER
void nbnxn_kokkos_finalize() KOKKOS_FUNC_TERM

/** Initializes atom-data for the Kokkos, called at every pair search step. */
KOKKOS_FUNC_QUALIFIER
void nbnxn_kokkos_init_atomdata(gmx_nbnxn_kokkos_t gmx_unused            *kknb,
				const struct nbnxn_atomdata_t gmx_unused *nbat) KOKKOS_FUNC_TERM

#ifdef __cplusplus
}
#endif

#endif /* NBNXN_KOKKOS_DATA_MGMT_H */