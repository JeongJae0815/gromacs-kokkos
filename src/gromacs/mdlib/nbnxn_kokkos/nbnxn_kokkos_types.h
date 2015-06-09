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
 *  Data types used internally in the nbnxn_kokkos module.
 *
 *  \author Sikandar Y. Mashayak <symashayak@gmail.com>
 *  \ingroup module_mdlib
 */

#ifndef NBNXN_KOKKOS_TYPES_H
#define NBNXN_KOKKOS_TYPES_H

#include "config.h"

#include <impl/Kokkos_Timer.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_Vectorization.hpp>

#include "gromacs/gmxlib/kokkos_tools/kokkos_type.h"
#include "gromacs/legacyheaders/types/interaction_const.h"
#include "gromacs/mdlib/nbnxn_pairlist.h"

#ifdef __cplusplus
extern "C" {
#endif

/* All structs prefixed with "kokkos_" hold data used in Kokkos calculations and
 * are passed to the kernels, except kokkos_timers_t. */
/*! \cond */
typedef struct kokkos_atomdata  kokkos_atomdata_t;
/*! \endcond */

/** \internal
 * \brief Nonbonded atom data - both inputs and outputs.
 */
struct kokkos_atomdata
{
    int      natoms;            /**< number of atoms                              */
    int      natoms_local;      /**< number of local atoms                        */
    int      nalloc;            /**< allocation size for the atom data (xq, f)    */

    int      ntype;             /**< number of atom types                         */

    bool     bShiftVecUploaded; /**< true if the shift vector has been uploaded   */

 /* Dual Kokkos views */
    DAT::tdual_xq_array   k_xq;     /**< atom coordinates + charges, size natoms kokkos dual view */
    DAT::tdual_f_array    k_f;      /**< force output array, size natoms kokkos dual view         */

    DAT::tdual_real_1d   k_e_lj;   /**< LJ energy output, size 1 kokkos dual view                */
    DAT::tdual_real_1d   k_e_el;   /**< Electrostatics energy input, size 1 kokkos dual view     */

    DAT::tdual_real_1d    k_fshift; /**< shift forces kokkos dual view                            */


    DAT::tdual_int_1d k_atom_types; /**< atom type indices, size natoms kokkos dual view          */

    DAT::tdual_real_1d_3  k_shift_vec;/**< shifts kokkos dual view                               */

  /* Views on Kokkos device */
    DAT::t_xq_array   d_xq;     /**< atom coordinates + charges, size natoms on kokkos device */
    DAT::t_f_array    d_f;      /**< force output array, size natoms on kokkos device         */

    DAT::t_real_1d   d_e_lj;   /**< LJ energy output, size 1 on kokkos device                */
    DAT::t_real_1d   d_e_el;   /**< Electrostatics energy input, size 1 on kokkos device     */

    DAT::t_real_1d    d_fshift; /**< shift forces on kokkos device                            */


    DAT::t_int_1d d_atom_types; /**< atom type indices, size natoms on kokkos device          */

    DAT::t_real_1d_3  d_shift_vec;/**< shifts on kokkos device                                  */

  /* Views on Kokkos host */
    HAT::t_xq_array   h_xq;     /**< atom coordinates + charges, size natoms on kokkos host  */
    HAT::t_f_array    h_f;      /**< force output array, size natoms on kokkos host          */

    HAT::t_real_1d   h_e_lj;   /**< LJ energy output, size 1 on kokkos host                 */
    HAT::t_real_1d   h_e_el;   /**< Electrostatics energy input, size 1 on kokkos host      */

    HAT::t_real_1d    h_fshift; /**< shift forces on kokkos host                             */


    HAT::t_int_1d h_atom_types; /**< atom type indices, size natoms on kokkos host           */

    HAT::t_real_1d_3  h_shift_vec;/**< shifts on kokkos host                                   */

};

/** \internal
 * \brief Main data structure for Kokkos nonbonded force calculations.
 */
struct nbnxn_kokkos_t
{
  kokkos_atomdata_t            *atdat;          /**< atom data                                            */
};

#ifdef __cplusplus
}
#endif

#endif  /* NBNXN_KOKKOS_TYPES_H */
