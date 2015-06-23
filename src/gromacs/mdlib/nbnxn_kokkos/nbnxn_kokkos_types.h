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
 *  Data types used for the Kokkos kernel.
 *
 *  \author Sikandar Y. Mashayak <symashayak@gmail.com>
 *  \ingroup module_mdlib
 */

#ifndef NBNXN_KOKKOS_TYPES_H
#define NBNXN_KOKKOS_TYPES_H

#include "config.h"

#include "gromacs/gmxlib/kokkos_tools/kokkos_type.h"
#include "gromacs/mdlib/nbnxn_kokkos_types.h"

#ifdef __cplusplus
extern "C" {
#endif

    struct kokkos_atomdata_t
    {
        DAT::tdual_real_1d       k_x;         /* dual view for x              */
        DAT::t_real_1d           d_x;         /* device view for x            */
        HAT::t_real_1d           h_x;         /* host view for x              */
    };

    struct kokkos_pairlist_t
    {
    
        DAT::tdual_ci_1d       k_ci;          /* The i-cluster list, size nci             */
        DAT::tdual_cj_1d       k_cj;          /* The j-cluster list, size ncj             */

        DAT::t_ci_1d           d_ci;          /* The i-cluster list, size nci             */
        DAT::t_cj_1d           d_cj;          /* The j-cluster list, size ncj             */

        HAT::t_ci_1d           h_ci;          /* The i-cluster list, size nci             */
        HAT::t_cj_1d           h_cj;          /* The j-cluster list, size ncj             */

        // ci and cj arrays are allocated inside #pragma omp loops and 
        // Kokkos does not allow allocating/managing views from within parallel loops
        // so for now using unmanaged views for ci and cj
        HAT::t_un_ci_1d        h_un_ci;       /* unmanaged host view for i-cluster list   */
        HAT::t_un_cj_1d        h_un_cj;       /* unmanaged host view for j-cluster list   */

    };

#ifdef __cplusplus
}
#endif

#endif  /* NBNXN_KOKKOS_TYPES_H */
