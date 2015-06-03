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

/*! \internal \file
 *  \brief
 *  Data types used internally in the nbnxn_kokkos module.
 *
 *  \author Sikandar Y. Mashayak <symashayak@gmail.com>
 *  \ingroup module_mdlib
 */

#ifndef GMX_KOKKOS_TYPE_H
#define GMX_KOKKOS_TYPE_H

#include "config.h"

#include <impl/Kokkos_Timer.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_Vectorization.hpp>

// set GMXHostype and GMXDeviceType from Kokkos Default Types
typedef Kokkos::DefaultExecutionSpace GMXDeviceType;
typedef Kokkos::HostSpace::execution_space GMXHostType;

// set ExecutionSpace stuct with variable "space"

// template<class Device>
// struct ExecutionSpaceFromDevice;

// template<>
// struct ExecutionSpaceFromDevice<LMPHostType> {
//   static const LAMMPS_NS::ExecutionSpace space = LAMMPS_NS::Host;
// };
// #ifdef KOKKOS_HAVE_CUDA
// template<>
// struct ExecutionSpaceFromDevice<Kokkos::Cuda> {
//   static const LAMMPS_NS::ExecutionSpace space = LAMMPS_NS::Device;
// };
// #endif

typedef float GMX_FLOAT;
typedef float X_FLOAT;
typedef float XQ_FLOAT;
typedef float V_FLOAT;
typedef float F_FLOAT;
typedef float E_FLOAT;

// GROMACS types

template <class DeviceType>
struct ArrayTypes;

template <>
struct ArrayTypes<GMXDeviceType> {

// scalar types

typedef Kokkos::
  DualView<int, GMXDeviceType::array_layout, GMXDeviceType> tdual_int_scalar;
typedef tdual_int_scalar::t_dev t_int_scalar;
typedef tdual_int_scalar::t_dev_const t_int_scalar_const;
typedef tdual_int_scalar::t_dev_um t_int_scalar_um;
typedef tdual_int_scalar::t_dev_const_um t_int_scalar_const_um;

typedef Kokkos::
  DualView<GMX_FLOAT, GMXDeviceType::array_layout, GMXDeviceType> 
  tdual_float_scalar;
typedef tdual_float_scalar::t_dev t_float_scalar;
typedef tdual_float_scalar::t_dev_const t_float_scalar_const;
typedef tdual_float_scalar::t_dev_um t_float_scalar_um;
typedef tdual_float_scalar::t_dev_const_um t_float_scalar_const_um;

// generic array types

typedef Kokkos::
  DualView<int*, GMXDeviceType::array_layout, GMXDeviceType> tdual_int_1d;
typedef tdual_int_1d::t_dev t_int_1d;
typedef tdual_int_1d::t_dev_const t_int_1d_const;
typedef tdual_int_1d::t_dev_um t_int_1d_um;
typedef tdual_int_1d::t_dev_const_um t_int_1d_const_um;
typedef tdual_int_1d::t_dev_const_randomread t_int_1d_randomread;

// 1d float array n
typedef Kokkos::DualView<GMX_FLOAT*, GMXDeviceType::array_layout, GMXDeviceType> tdual_float_1d;
typedef tdual_float_1d::t_dev t_float_1d;
typedef tdual_float_1d::t_dev_const t_float_1d_const;
typedef tdual_float_1d::t_dev_um t_float_1d_um;
typedef tdual_float_1d::t_dev_const_um t_float_1d_const_um;
typedef tdual_float_1d::t_dev_const_randomread t_float_1d_randomread;

//2d float array n
typedef Kokkos::DualView<GMX_FLOAT**, GMXDeviceType::array_layout, GMXDeviceType> tdual_float_2d;
typedef tdual_float_2d::t_dev t_float_2d;
typedef tdual_float_2d::t_dev_const t_float_2d_const;
typedef tdual_float_2d::t_dev_um t_float_2d_um;
typedef tdual_float_2d::t_dev_const_um t_float_2d_const_um;
typedef tdual_float_2d::t_dev_const_randomread t_float_2d_randomread;

//2d float array n*3 
typedef Kokkos::DualView<GMX_FLOAT*[3], GMXDeviceType::array_layout, GMXDeviceType> tdual_f_array;
typedef tdual_f_array::t_dev t_f_array;
typedef tdual_f_array::t_dev_const t_f_array_const;
typedef tdual_f_array::t_dev_um t_f_array_um;
typedef tdual_f_array::t_dev_const_um t_f_array_const_um;
typedef tdual_f_array::t_dev_const_randomread t_f_array_randomread;

//2d float array n*4 
typedef Kokkos::DualView<GMX_FLOAT*[4], GMXDeviceType::array_layout, GMXDeviceType> tdual_xq_array;
typedef tdual_xq_array::t_dev t_xq_array;
typedef tdual_xq_array::t_dev_const t_xq_array_const;
typedef tdual_xq_array::t_dev_um t_xq_array_um;
typedef tdual_xq_array::t_dev_const_um t_xq_array_const_um;
typedef tdual_xq_array::t_dev_const_randomread t_xq_array_randomread;

};

#ifdef KOKKOS_HAVE_CUDA
template <>
struct ArrayTypes<GMXHostType> {

// scalar types

typedef Kokkos::
  DualView<int, GMXDeviceType::array_layout, GMXDeviceType> tdual_int_scalar;
typedef tdual_int_scalar::t_host t_int_scalar;
typedef tdual_int_scalar::t_host_const t_int_scalar_const;
typedef tdual_int_scalar::t_host_um t_int_scalar_um;
typedef tdual_int_scalar::t_host_const_um t_int_scalar_const_um;

typedef Kokkos::
  DualView<GMX_FLOAT, GMXDeviceType::array_layout, GMXDeviceType> 
  tdual_float_scalar;
typedef tdual_float_scalar::t_host t_float_scalar;
typedef tdual_float_scalar::t_host_const t_float_scalar_const;
typedef tdual_float_scalar::t_host_um t_float_scalar_um;
typedef tdual_float_scalar::t_host_const_um t_float_scalar_const_um;

// generic array types

typedef Kokkos::
  DualView<int*, GMXDeviceType::array_layout, GMXDeviceType> tdual_int_1d;
typedef tdual_int_1d::t_host t_int_1d;
typedef tdual_int_1d::t_host_const t_int_1d_const;
typedef tdual_int_1d::t_host_um t_int_1d_um;
typedef tdual_int_1d::t_host_const_um t_int_1d_const_um;
typedef tdual_int_1d::t_host_const_randomread t_int_1d_randomread;

// 1d float array n
typedef Kokkos::DualView<GMX_FLOAT*, GMXDeviceType::array_layout, GMXDeviceType> tdual_float_1d;
typedef tdual_float_1d::t_host t_float_1d;
typedef tdual_float_1d::t_host_const t_float_1d_const;
typedef tdual_float_1d::t_host_um t_float_1d_um;
typedef tdual_float_1d::t_host_const_um t_float_1d_const_um;
typedef tdual_float_1d::t_host_const_randomread t_float_1d_randomread;

//2d float array n
typedef Kokkos::DualView<GMX_FLOAT**, GMXDeviceType::array_layout, GMXDeviceType> tdual_float_2d;
typedef tdual_float_2d::t_host t_float_2d;
typedef tdual_float_2d::t_host_const t_float_2d_const;
typedef tdual_float_2d::t_host_um t_float_2d_um;
typedef tdual_float_2d::t_host_const_um t_float_2d_const_um;
typedef tdual_float_2d::t_host_const_randomread t_float_2d_randomread;

//2d float array n*3 
typedef Kokkos::DualView<GMX_FLOAT*[3], GMXDeviceType::array_layout, GMXDeviceType> tdual_f_array;
typedef tdual_f_array::t_host t_f_array;
typedef tdual_f_array::t_host_const t_f_array_const;
typedef tdual_f_array::t_host_um t_f_array_um;
typedef tdual_f_array::t_host_const_um t_f_array_const_um;
typedef tdual_f_array::t_host_const_randomread t_f_array_randomread;

//2d float array n*4 
typedef Kokkos::DualView<GMX_FLOAT*[4], GMXDeviceType::array_layout, GMXDeviceType> tdual_xq_array;
typedef tdual_xq_array::t_host t_xq_array;
typedef tdual_xq_array::t_host_const t_xq_array_const;
typedef tdual_xq_array::t_host_um t_xq_array_um;
typedef tdual_xq_array::t_host_const_um t_xq_array_const_um;
typedef tdual_xq_array::t_host_const_randomread t_xq_array_randomread;

};
#endif /* KOKKOS_HAVE_CUDA */

//default Gromacs Types
typedef struct ArrayTypes<GMXDeviceType> DAT;
typedef struct ArrayTypes<GMXHostType> HAT;

template<class DeviceType>
struct MemsetZeroFunctor {
  typedef DeviceType  device_type ;
  void* ptr;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    ((int*)ptr)[i] = 0;
  }
};

template<class ViewType>
void memset_kokkos (ViewType &view) {
  static MemsetZeroFunctor<typename ViewType::device_type> f;
  f.ptr = view.ptr_on_device();
  Kokkos::parallel_for(view.capacity()*sizeof(typename ViewType::value_type)/4, f);
  ViewType::device_type::fence();
}

#endif /* GMX_KOKKOS_TYPE_H */
