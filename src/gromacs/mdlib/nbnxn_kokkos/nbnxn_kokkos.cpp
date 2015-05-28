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
 *  \brief Define CUDA implementation of nbnxn_gpu.h
 *
 *  \author Szilard Pall <pall.szilard@gmail.com>
 */
#include "gmxpre.h"

#include "config.h"

#include <assert.h>
#include <stdlib.h>

#include "gromacs/mdlib/nbnxn_gpu.h"

#if defined(_MSVC)
#include <limits>
#endif

#include <cuda.h>

#ifdef TMPI_ATOMICS
#include "thread_mpi/atomic.h"
#endif

#include "gromacs/gmxlib/cuda_tools/cudautils.cuh"
#include "gromacs/legacyheaders/types/force_flags.h"
#include "gromacs/legacyheaders/types/simple.h"
#include "gromacs/mdlib/nb_verlet.h"
#include "gromacs/mdlib/nbnxn_consts.h"
#include "gromacs/mdlib/nbnxn_gpu_data_mgmt.h"
#include "gromacs/mdlib/nbnxn_pairlist.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/timing/gpu_timing.h"
#include "gromacs/utility/cstringutil.h"

#include "nbnxn_cuda_types.h"

#if defined HAVE_CUDA_TEXOBJ_SUPPORT && __CUDA_ARCH__ >= 300
#define USE_TEXOBJ
#endif

/*! Texture reference for LJ C6/C12 parameters; bound to cu_nbparam_t.nbfp */
texture<float, 1, cudaReadModeElementType> nbfp_texref;

/*! Texture reference for LJ-PME parameters; bound to cu_nbparam_t.nbfp_comb */
texture<float, 1, cudaReadModeElementType> nbfp_comb_texref;

/*! Texture reference for Ewald coulomb force table; bound to cu_nbparam_t.coulomb_tab */
texture<float, 1, cudaReadModeElementType> coulomb_tab_texref;

/* Convenience defines */
#define NCL_PER_SUPERCL         (NBNXN_GPU_NCLUSTER_PER_SUPERCLUSTER)
#define CL_SIZE                 (NBNXN_GPU_CLUSTER_SIZE)

/* NTHREAD_Z controls the number of j-clusters processed concurrently on NTHREAD_Z
 * warp-pairs per block.
 *
 * - On CC 2.0-3.5, 5.0, and 5.2, NTHREAD_Z == 1, translating to 64 th/block with 16
 * blocks/multiproc, is the fastest even though this setup gives low occupancy.
 * NTHREAD_Z > 1 results in excessive register spilling unless the minimum blocks
 * per multiprocessor is reduced proportionally to get the original number of max
 * threads in flight (and slightly lower performance).
 * - On CC 3.7 there are enough registers to double the number of threads; using
 * NTHREADS_Z == 2 is fastest with 16 blocks (TODO: test with RF and other kernels
 * with low-register use).
 *
 * Note that the current kernel implementation only supports NTHREAD_Z > 1 with
 * shuffle-based reduction, hence CC >= 3.0.
 */

/* Kernel launch bounds as function of NTHREAD_Z.
 * - CC 3.5/5.2: NTHREAD_Z=1, (64, 16) bounds
 * - CC 3.7:     NTHREAD_Z=2, (128, 16) bounds
 */
#if __CUDA_ARCH__ == 370
#define NTHREAD_Z           (2)
#define MIN_BLOCKS_PER_MP   (16)
#else
#define NTHREAD_Z           (1)
#define MIN_BLOCKS_PER_MP   (16)
#endif
#define THREADS_PER_BLOCK   (CL_SIZE*CL_SIZE*NTHREAD_Z)


/***** The kernels come here *****/
#include "gromacs/mdlib/nbnxn_cuda/nbnxn_cuda_kernel_utils.cuh"

/* Top-level kernel generation: will generate through multiple inclusion the
 * following flavors for all kernels:
 * - force-only output;
 * - force and energy output;
 * - force-only with pair list pruning;
 * - force and energy output with pair list pruning.
 */
/** Force only **/
#include "gromacs/mdlib/nbnxn_cuda/nbnxn_cuda_kernels.cuh"
/** Force & energy **/
#define CALC_ENERGIES
#include "gromacs/mdlib/nbnxn_cuda/nbnxn_cuda_kernels.cuh"
#undef CALC_ENERGIES

/*** Pair-list pruning kernels ***/
/** Force only **/
#define PRUNE_NBL
#include "gromacs/mdlib/nbnxn_cuda/nbnxn_cuda_kernels.cuh"
/** Force & energy **/
#define CALC_ENERGIES
#include "gromacs/mdlib/nbnxn_cuda/nbnxn_cuda_kernels.cuh"
#undef CALC_ENERGIES
#undef PRUNE_NBL


/*! As we execute nonbonded workload in separate streams, before
   launching the kernel we need to make sure that the following
   operations have completed: - atomdata allocation and related H2D
   transfers (every nstlist step); - pair list H2D transfer (every
   nstlist step); - shift vector H2D transfer (every nstlist step); -
   force (+shift force and energy) output clearing (every step).

   These operations are issued in the local stream at the beginning of
   the step and therefore always complete before the local kernel
   launch. The non-local kernel is launched after the local on the
   same device/context, so this is inherently scheduled after the
   operations in the local stream (including the above "misc_ops").
   However, for the sake of having a future-proof implementation, we
   use the misc_ops_done event to record the point in time when the
   above operations are finished and synchronize with this event in
   the non-local stream.
 */
void nbnxn_gpu_launch_kernel()
{

}

void nbnxn_gpu_launch_cpyback()
{
}
