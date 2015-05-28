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

#include "nbnxn_kokkos_data_mgmt.h"

#include "config.h"

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "gromacs/legacyheaders/typedefs.h"
#include "gromacs/legacyheaders/types/enums.h"

#include "gromacs/pbcutil/ishift.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/smalloc.h"

void nbnxn_kokkos_init()
{

  int device = 0;
  int num_threads = 1;
  int numa = 1;

#ifdef KOKKOS_HAVE_CUDA
  Kokkos::HostSpace::execution_space::initialize(num_threads,numa);
  Kokkos::Cuda::SelectDevice select_device(device);
  Kokkos::Cuda::initialize(select_device);
#else
  GMXHostType::initialize(num_threads,numa);
#endif

}

void nbnxn_kokkos_finalize()
{

#ifdef KOKKOS_HAVE_CUDA
  Kokkos::Cuda::finalize();
  Kokkos::HostSpace::execution_space::finalize();
#else
  GMXHostType::finalize();
#endif

}
