/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2013,2014,  by the GROMACS development team, led by
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

/* Note that floating-point constants in CUDA code should be suffixed
 * with f (e.g. 0.5f), to stop the compiler producing intermediate
 * code that is in double precision.
 */
#include "config.h"

#ifndef NBNXN_KOKKOS_KERNEL_UTILS_CUH
#define NBNXN_KOKKOS_KERNEL_UTILS_CUH

#define ONE_SIXTH_F     0.16666667f
#define ONE_TWELVETH_F  0.08333333f


/*! Apply force switch,  force + energy version. */
static inline 
void calculate_force_switch_F(float               c6,
                              float               c12,
                              float               inv_r,
                              float               r2,
                              float              *F_invr)
{
    float r, r_switch;

    r         = r2 * inv_r;
    r_switch  = r - nbparam.rvdw_switch;
    r_switch  = r_switch >= 0.0f ? r_switch : 0.0f;

    *F_invr  +=  -c6 +  c12*;

}

#endif /* NBNXN_KOKKOS_KERNEL_UTILS_CUH */
