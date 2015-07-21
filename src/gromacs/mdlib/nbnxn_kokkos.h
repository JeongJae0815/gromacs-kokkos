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

#ifndef GMX_MDLIB_NBNXN_KOKKOS_H
#define GMX_MDLIB_NBNXN_KOKKOS_H

#include "gromacs/gmxlib/kokkos_tools/kokkos_macros.h"
#include "gromacs/gmxlib/kokkos_tools/kokkos_type.h"
#include "gromacs/legacyheaders/types/interaction_const.h"
#include "gromacs/legacyheaders/types/simple.h"
#include "gromacs/mdlib/nbnxn_pairlist.h"

#ifdef __cplusplus
extern "C" {
#endif

    KOKKOS_FUNC_QUALIFIER
    void nbnxn_kokkos_launch_kernel(nbnxn_pairlist_set_t      gmx_unused *nbl_list,
                                    nbnxn_atomdata_t    gmx_unused *nbat,
                                    const interaction_const_t gmx_unused *ic,
                                    int                       gmx_unused  ewald_excl,
                                    rvec                      gmx_unused *shift_vec,
                                    int                       gmx_unused  force_flags,
                                    int                       gmx_unused  clearF,
                                    real                      gmx_unused *fshift,
                                    real                      gmx_unused *Vc,
                                    real                      gmx_unused *Vvdw) KOKKOS_FUNC_TERM
    
#ifdef __cplusplus
        }
#endif


#endif
