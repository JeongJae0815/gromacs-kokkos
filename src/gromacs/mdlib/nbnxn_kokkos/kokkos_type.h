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

#ifndef LMP_LMPTYPE_KOKKOS_H
#define LMP_LMPTYPE_KOKKOS_H

#include "config.h"

#include <impl/Kokkos_Timer.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_Vectorization.hpp>

#if !defined(__CUDACC__) && !defined(__VECTOR_TYPES_H__)
  struct double2 {
    double x, y;
  };
  struct float2 {
    float x, y;
  };
  struct double4 {
    double x, y, z, w;
  };
  struct float4 {
    float x, y, z, w;
  };
#endif

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

typedef Kokkos::
  DualView<int*[3], Kokkos::LayoutRight, GMXDeviceType> tdual_int_1d_3;
typedef tdual_int_1d_3::t_dev t_int_1d_3;
typedef tdual_int_1d_3::t_dev_const t_int_1d_3_const;
typedef tdual_int_1d_3::t_dev_um t_int_1d_3_um;
typedef tdual_int_1d_3::t_dev_const_um t_int_1d_3_const_um;
typedef tdual_int_1d_3::t_dev_const_randomread t_int_1d_3_randomread;

typedef Kokkos::
  DualView<int**, Kokkos::LayoutRight, GMXDeviceType> tdual_int_2d;
typedef tdual_int_2d::t_dev t_int_2d;
typedef tdual_int_2d::t_dev_const t_int_2d_const;
typedef tdual_int_2d::t_dev_um t_int_2d_um;
typedef tdual_int_2d::t_dev_const_um t_int_2d_const_um;
typedef tdual_int_2d::t_dev_const_randomread t_int_2d_randomread;

typedef Kokkos::
  DualView<double*, Kokkos::LayoutRight, GMXDeviceType> tdual_double_1d;
typedef tdual_double_1d::t_dev t_double_1d;
typedef tdual_double_1d::t_dev_const t_double_1d_const;
typedef tdual_double_1d::t_dev_um t_double_1d_um;
typedef tdual_double_1d::t_dev_const_um t_double_1d_const_um;
typedef tdual_double_1d::t_dev_const_randomread t_double_1d_randomread;

typedef Kokkos::
  DualView<double**, Kokkos::LayoutRight, GMXDeviceType> tdual_double_2d;
typedef tdual_double_2d::t_dev t_double_2d;
typedef tdual_double_2d::t_dev_const t_double_2d_const;
typedef tdual_double_2d::t_dev_um t_double_2d_um;
typedef tdual_double_2d::t_dev_const_um t_double_2d_const_um;
typedef tdual_double_2d::t_dev_const_randomread t_double_2d_randomread;

// 1d float array n

typedef Kokkos::DualView<GMX_FLOAT*, GMXDeviceType::array_layout, GMXDeviceType> tdual_float_1d;
typedef tdual_float_1d::t_dev t_float_1d;
typedef tdual_float_1d::t_dev_const t_float_1d_const;
typedef tdual_float_1d::t_dev_um t_float_1d_um;
typedef tdual_float_1d::t_dev_const_um t_float_1d_const_um;
typedef tdual_float_1d::t_dev_const_randomread t_float_1d_randomread;

//2d float array n
typedef Kokkos::DualView<GMX_FLOAT**, Kokkos::LayoutRight, GMXDeviceType> tdual_float_2d;
typedef tdual_float_2d::t_dev t_float_2d;
typedef tdual_float_2d::t_dev_const t_float_2d_const;
typedef tdual_float_2d::t_dev_um t_float_2d_um;
typedef tdual_float_2d::t_dev_const_um t_float_2d_const_um;
typedef tdual_float_2d::t_dev_const_randomread t_float_2d_randomread;

//Position Types
//1d X_FLOAT array n
typedef Kokkos::DualView<X_FLOAT*, GMXDeviceType::array_layout, GMXDeviceType> tdual_xfloat_1d;
typedef tdual_xfloat_1d::t_dev t_xfloat_1d;
typedef tdual_xfloat_1d::t_dev_const t_xfloat_1d_const;
typedef tdual_xfloat_1d::t_dev_um t_xfloat_1d_um;
typedef tdual_xfloat_1d::t_dev_const_um t_xfloat_1d_const_um;
typedef tdual_xfloat_1d::t_dev_const_randomread t_xfloat_1d_randomread;

//2d X_FLOAT array n*m
typedef Kokkos::DualView<X_FLOAT**, Kokkos::LayoutRight, GMXDeviceType> tdual_xfloat_2d;
typedef tdual_xfloat_2d::t_dev t_xfloat_2d;
typedef tdual_xfloat_2d::t_dev_const t_xfloat_2d_const;
typedef tdual_xfloat_2d::t_dev_um t_xfloat_2d_um;
typedef tdual_xfloat_2d::t_dev_const_um t_xfloat_2d_const_um;
typedef tdual_xfloat_2d::t_dev_const_randomread t_xfloat_2d_randomread;

//2d X_FLOAT array n*4 
#ifdef GMX_KOKKOS_NO_LEGACY
typedef Kokkos::DualView<X_FLOAT*[3], Kokkos::LayoutLeft, GMXDeviceType> tdual_x_array;
#else
typedef Kokkos::DualView<X_FLOAT*[3], Kokkos::LayoutRight, GMXDeviceType> tdual_x_array;
#endif
typedef tdual_x_array::t_dev t_x_array;
typedef tdual_x_array::t_dev_const t_x_array_const;
typedef tdual_x_array::t_dev_um t_x_array_um;
typedef tdual_x_array::t_dev_const_um t_x_array_const_um;
typedef tdual_x_array::t_dev_const_randomread t_x_array_randomread;

//Velocity Types
//1d V_FLOAT array n
typedef Kokkos::DualView<V_FLOAT*, GMXDeviceType::array_layout, GMXDeviceType> tdual_vfloat_1d;
typedef tdual_vfloat_1d::t_dev t_vfloat_1d;
typedef tdual_vfloat_1d::t_dev_const t_vfloat_1d_const;
typedef tdual_vfloat_1d::t_dev_um t_vfloat_1d_um;
typedef tdual_vfloat_1d::t_dev_const_um t_vfloat_1d_const_um;
typedef tdual_vfloat_1d::t_dev_const_randomread t_vfloat_1d_randomread;

//2d V_FLOAT array n*m
typedef Kokkos::DualView<V_FLOAT**, Kokkos::LayoutRight, GMXDeviceType> tdual_vfloat_2d;
typedef tdual_vfloat_2d::t_dev t_vfloat_2d;
typedef tdual_vfloat_2d::t_dev_const t_vfloat_2d_const;
typedef tdual_vfloat_2d::t_dev_um t_vfloat_2d_um;
typedef tdual_vfloat_2d::t_dev_const_um t_vfloat_2d_const_um;
typedef tdual_vfloat_2d::t_dev_const_randomread t_vfloat_2d_randomread;

//2d V_FLOAT array n*3
typedef Kokkos::DualView<V_FLOAT*[3], Kokkos::LayoutRight, GMXDeviceType> tdual_v_array;
//typedef Kokkos::DualView<V_FLOAT*[3], GMXDeviceType::array_layout, GMXDeviceType> tdual_v_array;
typedef tdual_v_array::t_dev t_v_array;
typedef tdual_v_array::t_dev_const t_v_array_const;
typedef tdual_v_array::t_dev_um t_v_array_um;
typedef tdual_v_array::t_dev_const_um t_v_array_const_um;
typedef tdual_v_array::t_dev_const_randomread t_v_array_randomread;

//Force Types
//1d F_FLOAT array n

typedef Kokkos::DualView<F_FLOAT*, GMXDeviceType::array_layout, GMXDeviceType> tdual_ffloat_1d;
typedef tdual_ffloat_1d::t_dev t_ffloat_1d;
typedef tdual_ffloat_1d::t_dev_const t_ffloat_1d_const;
typedef tdual_ffloat_1d::t_dev_um t_ffloat_1d_um;
typedef tdual_ffloat_1d::t_dev_const_um t_ffloat_1d_const_um;
typedef tdual_ffloat_1d::t_dev_const_randomread t_ffloat_1d_randomread;

//2d F_FLOAT array n*m

typedef Kokkos::DualView<F_FLOAT**, Kokkos::LayoutRight, GMXDeviceType> tdual_ffloat_2d;
typedef tdual_ffloat_2d::t_dev t_ffloat_2d;
typedef tdual_ffloat_2d::t_dev_const t_ffloat_2d_const;
typedef tdual_ffloat_2d::t_dev_um t_ffloat_2d_um;
typedef tdual_ffloat_2d::t_dev_const_um t_ffloat_2d_const_um;
typedef tdual_ffloat_2d::t_dev_const_randomread t_ffloat_2d_randomread;

//2d F_FLOAT array n*3

typedef Kokkos::DualView<F_FLOAT*[3], Kokkos::LayoutRight, GMXDeviceType> tdual_f_array;
//typedef Kokkos::DualView<F_FLOAT*[3], GMXDeviceType::array_layout, GMXDeviceType> tdual_f_array;
typedef tdual_f_array::t_dev t_f_array;
typedef tdual_f_array::t_dev_const t_f_array_const;
typedef tdual_f_array::t_dev_um t_f_array_um;
typedef tdual_f_array::t_dev_const_um t_f_array_const_um;
typedef tdual_f_array::t_dev_const_randomread t_f_array_randomread;

//2d F_FLOAT array n*6 (for virial)

typedef Kokkos::DualView<F_FLOAT*[6], Kokkos::LayoutRight, GMXDeviceType> tdual_virial_array;
typedef tdual_virial_array::t_dev t_virial_array;
typedef tdual_virial_array::t_dev_const t_virial_array_const;
typedef tdual_virial_array::t_dev_um t_virial_array_um;
typedef tdual_virial_array::t_dev_const_um t_virial_array_const_um;
typedef tdual_virial_array::t_dev_const_randomread t_virial_array_randomread;

//Energy Types
//1d E_FLOAT array n

typedef Kokkos::DualView<E_FLOAT*, GMXDeviceType::array_layout, GMXDeviceType> tdual_efloat_1d;
typedef tdual_efloat_1d::t_dev t_efloat_1d;
typedef tdual_efloat_1d::t_dev_const t_efloat_1d_const;
typedef tdual_efloat_1d::t_dev_um t_efloat_1d_um;
typedef tdual_efloat_1d::t_dev_const_um t_efloat_1d_const_um;
typedef tdual_efloat_1d::t_dev_const_randomread t_efloat_1d_randomread;

//2d E_FLOAT array n*m

typedef Kokkos::DualView<E_FLOAT**, Kokkos::LayoutRight, GMXDeviceType> tdual_efloat_2d;
typedef tdual_efloat_2d::t_dev t_efloat_2d;
typedef tdual_efloat_2d::t_dev_const t_efloat_2d_const;
typedef tdual_efloat_2d::t_dev_um t_efloat_2d_um;
typedef tdual_efloat_2d::t_dev_const_um t_efloat_2d_const_um;
typedef tdual_efloat_2d::t_dev_const_randomread t_efloat_2d_randomread;

//2d E_FLOAT array n*3 

typedef Kokkos::DualView<E_FLOAT*[3], Kokkos::LayoutRight, GMXDeviceType> tdual_e_array;
typedef tdual_e_array::t_dev t_e_array;
typedef tdual_e_array::t_dev_const t_e_array_const;
typedef tdual_e_array::t_dev_um t_e_array_um;
typedef tdual_e_array::t_dev_const_um t_e_array_const_um;
typedef tdual_e_array::t_dev_const_randomread t_e_array_randomread;

//Neighbor Types

typedef Kokkos::DualView<int**, GMXDeviceType::array_layout, GMXDeviceType> tdual_neighbors_2d;
typedef tdual_neighbors_2d::t_dev t_neighbors_2d;
typedef tdual_neighbors_2d::t_dev_const t_neighbors_2d_const;
typedef tdual_neighbors_2d::t_dev_um t_neighbors_2d_um;
typedef tdual_neighbors_2d::t_dev_const_um t_neighbors_2d_const_um;
typedef tdual_neighbors_2d::t_dev_const_randomread t_neighbors_2d_randomread;

};

#ifdef KOKKOS_HAVE_CUDA
template <>
struct ArrayTypes<GMXHostType> {

//Scalar Types

typedef Kokkos::DualView<int, GMXDeviceType::array_layout, GMXDeviceType> tdual_int_scalar;
typedef tdual_int_scalar::t_host t_int_scalar;
typedef tdual_int_scalar::t_host_const t_int_scalar_const;
typedef tdual_int_scalar::t_host_um t_int_scalar_um;
typedef tdual_int_scalar::t_host_const_um t_int_scalar_const_um;

typedef Kokkos::DualView<GMX_FLOAT, GMXDeviceType::array_layout, GMXDeviceType> tdual_float_scalar;
typedef tdual_float_scalar::t_host t_float_scalar;
typedef tdual_float_scalar::t_host_const t_float_scalar_const;
typedef tdual_float_scalar::t_host_um t_float_scalar_um;
typedef tdual_float_scalar::t_host_const_um t_float_scalar_const_um;

//Generic ArrayTypes
typedef Kokkos::DualView<int*, GMXDeviceType::array_layout, GMXDeviceType> tdual_int_1d;
typedef tdual_int_1d::t_host t_int_1d;
typedef tdual_int_1d::t_host_const t_int_1d_const;
typedef tdual_int_1d::t_host_um t_int_1d_um;
typedef tdual_int_1d::t_host_const_um t_int_1d_const_um;
typedef tdual_int_1d::t_host_const_randomread t_int_1d_randomread;

typedef Kokkos::DualView<int*[3], Kokkos::LayoutRight, GMXDeviceType> tdual_int_1d_3;
typedef tdual_int_1d_3::t_host t_int_1d_3;
typedef tdual_int_1d_3::t_host_const t_int_1d_3_const;
typedef tdual_int_1d_3::t_host_um t_int_1d_3_um;
typedef tdual_int_1d_3::t_host_const_um t_int_1d_3_const_um;
typedef tdual_int_1d_3::t_host_const_randomread t_int_1d_3_randomread;

typedef Kokkos::DualView<int**, Kokkos::LayoutRight, GMXDeviceType> tdual_int_2d;
typedef tdual_int_2d::t_host t_int_2d;
typedef tdual_int_2d::t_host_const t_int_2d_const;
typedef tdual_int_2d::t_host_um t_int_2d_um;
typedef tdual_int_2d::t_host_const_um t_int_2d_const_um;
typedef tdual_int_2d::t_host_const_randomread t_int_2d_randomread;

typedef Kokkos::
  DualView<double*, Kokkos::LayoutRight, GMXDeviceType> tdual_double_1d;
typedef tdual_double_1d::t_host t_double_1d;
typedef tdual_double_1d::t_host_const t_double_1d_const;
typedef tdual_double_1d::t_host_um t_double_1d_um;
typedef tdual_double_1d::t_host_const_um t_double_1d_const_um;
typedef tdual_double_1d::t_host_const_randomread t_double_1d_randomread;

typedef Kokkos::
  DualView<double**, Kokkos::LayoutRight, GMXDeviceType> tdual_double_2d;
typedef tdual_double_2d::t_host t_double_2d;
typedef tdual_double_2d::t_host_const t_double_2d_const;
typedef tdual_double_2d::t_host_um t_double_2d_um;
typedef tdual_double_2d::t_host_const_um t_double_2d_const_um;
typedef tdual_double_2d::t_host_const_randomread t_double_2d_randomread;

//1d float array n
typedef Kokkos::DualView<GMX_FLOAT*, GMXDeviceType::array_layout, GMXDeviceType> tdual_float_1d;
typedef tdual_float_1d::t_host t_float_1d;
typedef tdual_float_1d::t_host_const t_float_1d_const;
typedef tdual_float_1d::t_host_um t_float_1d_um;
typedef tdual_float_1d::t_host_const_um t_float_1d_const_um;
typedef tdual_float_1d::t_host_const_randomread t_float_1d_randomread;

//2d float array n
typedef Kokkos::DualView<GMX_FLOAT**, Kokkos::LayoutRight, GMXDeviceType> tdual_float_2d;
typedef tdual_float_2d::t_host t_float_2d;
typedef tdual_float_2d::t_host_const t_float_2d_const;
typedef tdual_float_2d::t_host_um t_float_2d_um;
typedef tdual_float_2d::t_host_const_um t_float_2d_const_um;
typedef tdual_float_2d::t_host_const_randomread t_float_2d_randomread;

//Position Types
//1d X_FLOAT array n
typedef Kokkos::DualView<X_FLOAT*, GMXDeviceType::array_layout, GMXDeviceType> tdual_xfloat_1d;
typedef tdual_xfloat_1d::t_host t_xfloat_1d;
typedef tdual_xfloat_1d::t_host_const t_xfloat_1d_const;
typedef tdual_xfloat_1d::t_host_um t_xfloat_1d_um;
typedef tdual_xfloat_1d::t_host_const_um t_xfloat_1d_const_um;
typedef tdual_xfloat_1d::t_host_const_randomread t_xfloat_1d_randomread;

//2d X_FLOAT array n*m
typedef Kokkos::DualView<X_FLOAT**, Kokkos::LayoutRight, GMXDeviceType> tdual_xfloat_2d;
typedef tdual_xfloat_2d::t_host t_xfloat_2d;
typedef tdual_xfloat_2d::t_host_const t_xfloat_2d_const;
typedef tdual_xfloat_2d::t_host_um t_xfloat_2d_um;
typedef tdual_xfloat_2d::t_host_const_um t_xfloat_2d_const_um;
typedef tdual_xfloat_2d::t_host_const_randomread t_xfloat_2d_randomread;

//2d X_FLOAT array n*3
typedef Kokkos::DualView<X_FLOAT*[3], Kokkos::LayoutRight, GMXDeviceType> tdual_x_array;
typedef tdual_x_array::t_host t_x_array;
typedef tdual_x_array::t_host_const t_x_array_const;
typedef tdual_x_array::t_host_um t_x_array_um;
typedef tdual_x_array::t_host_const_um t_x_array_const_um;
typedef tdual_x_array::t_host_const_randomread t_x_array_randomread;

//Velocity Types
//1d V_FLOAT array n
typedef Kokkos::DualView<V_FLOAT*, GMXDeviceType::array_layout, GMXDeviceType> tdual_vfloat_1d;
typedef tdual_vfloat_1d::t_host t_vfloat_1d;
typedef tdual_vfloat_1d::t_host_const t_vfloat_1d_const;
typedef tdual_vfloat_1d::t_host_um t_vfloat_1d_um;
typedef tdual_vfloat_1d::t_host_const_um t_vfloat_1d_const_um;
typedef tdual_vfloat_1d::t_host_const_randomread t_vfloat_1d_randomread;

//2d V_FLOAT array n*m
typedef Kokkos::DualView<V_FLOAT**, Kokkos::LayoutRight, GMXDeviceType> tdual_vfloat_2d;
typedef tdual_vfloat_2d::t_host t_vfloat_2d;
typedef tdual_vfloat_2d::t_host_const t_vfloat_2d_const;
typedef tdual_vfloat_2d::t_host_um t_vfloat_2d_um;
typedef tdual_vfloat_2d::t_host_const_um t_vfloat_2d_const_um;
typedef tdual_vfloat_2d::t_host_const_randomread t_vfloat_2d_randomread;

//2d V_FLOAT array n*3
typedef Kokkos::DualView<V_FLOAT*[3], Kokkos::LayoutRight, GMXDeviceType> tdual_v_array;
//typedef Kokkos::DualView<V_FLOAT*[3], GMXDeviceType::array_layout, GMXDeviceType> tdual_v_array;
typedef tdual_v_array::t_host t_v_array;
typedef tdual_v_array::t_host_const t_v_array_const;
typedef tdual_v_array::t_host_um t_v_array_um;
typedef tdual_v_array::t_host_const_um t_v_array_const_um;
typedef tdual_v_array::t_host_const_randomread t_v_array_randomread;

//Force Types
//1d F_FLOAT array n
typedef Kokkos::DualView<F_FLOAT*, GMXDeviceType::array_layout, GMXDeviceType> tdual_ffloat_1d;
typedef tdual_ffloat_1d::t_host t_ffloat_1d;
typedef tdual_ffloat_1d::t_host_const t_ffloat_1d_const;
typedef tdual_ffloat_1d::t_host_um t_ffloat_1d_um;
typedef tdual_ffloat_1d::t_host_const_um t_ffloat_1d_const_um;
typedef tdual_ffloat_1d::t_host_const_randomread t_ffloat_1d_randomread;

//2d F_FLOAT array n*m
typedef Kokkos::DualView<F_FLOAT**, Kokkos::LayoutRight, GMXDeviceType> tdual_ffloat_2d;
typedef tdual_ffloat_2d::t_host t_ffloat_2d;
typedef tdual_ffloat_2d::t_host_const t_ffloat_2d_const;
typedef tdual_ffloat_2d::t_host_um t_ffloat_2d_um;
typedef tdual_ffloat_2d::t_host_const_um t_ffloat_2d_const_um;
typedef tdual_ffloat_2d::t_host_const_randomread t_ffloat_2d_randomread;

//2d F_FLOAT array n*3
typedef Kokkos::DualView<F_FLOAT*[3], Kokkos::LayoutRight, GMXDeviceType> tdual_f_array;
//typedef Kokkos::DualView<F_FLOAT*[3], GMXDeviceType::array_layout, GMXDeviceType> tdual_f_array;
typedef tdual_f_array::t_host t_f_array;
typedef tdual_f_array::t_host_const t_f_array_const;
typedef tdual_f_array::t_host_um t_f_array_um;
typedef tdual_f_array::t_host_const_um t_f_array_const_um;
typedef tdual_f_array::t_host_const_randomread t_f_array_randomread;

//2d F_FLOAT array n*6 (for virial)
typedef Kokkos::DualView<F_FLOAT*[6], Kokkos::LayoutRight, GMXDeviceType> tdual_virial_array;
typedef tdual_virial_array::t_host t_virial_array;
typedef tdual_virial_array::t_host_const t_virial_array_const;
typedef tdual_virial_array::t_host_um t_virial_array_um;
typedef tdual_virial_array::t_host_const_um t_virial_array_const_um;
typedef tdual_virial_array::t_host_const_randomread t_virial_array_randomread;



//Energy Types
//1d E_FLOAT array n
typedef Kokkos::DualView<E_FLOAT*, GMXDeviceType::array_layout, GMXDeviceType> tdual_efloat_1d;
typedef tdual_efloat_1d::t_host t_efloat_1d;
typedef tdual_efloat_1d::t_host_const t_efloat_1d_const;
typedef tdual_efloat_1d::t_host_um t_efloat_1d_um;
typedef tdual_efloat_1d::t_host_const_um t_efloat_1d_const_um;
typedef tdual_efloat_1d::t_host_const_randomread t_efloat_1d_randomread;

//2d E_FLOAT array n*m
typedef Kokkos::DualView<E_FLOAT**, Kokkos::LayoutRight, GMXDeviceType> tdual_efloat_2d;
typedef tdual_efloat_2d::t_host t_efloat_2d;
typedef tdual_efloat_2d::t_host_const t_efloat_2d_const;
typedef tdual_efloat_2d::t_host_um t_efloat_2d_um;
typedef tdual_efloat_2d::t_host_const_um t_efloat_2d_const_um;
typedef tdual_efloat_2d::t_host_const_randomread t_efloat_2d_randomread;

//2d E_FLOAT array n*3
typedef Kokkos::DualView<E_FLOAT*[3], Kokkos::LayoutRight, GMXDeviceType> tdual_e_array;
typedef tdual_e_array::t_host t_e_array;
typedef tdual_e_array::t_host_const t_e_array_const;
typedef tdual_e_array::t_host_um t_e_array_um;
typedef tdual_e_array::t_host_const_um t_e_array_const_um;
typedef tdual_e_array::t_host_const_randomread t_e_array_randomread;

//Neighbor Types
typedef Kokkos::DualView<int**, GMXDeviceType::array_layout, GMXDeviceType> tdual_neighbors_2d;
typedef tdual_neighbors_2d::t_host t_neighbors_2d;
typedef tdual_neighbors_2d::t_host_const t_neighbors_2d_const;
typedef tdual_neighbors_2d::t_host_um t_neighbors_2d_um;
typedef tdual_neighbors_2d::t_host_const_um t_neighbors_2d_const_um;
typedef tdual_neighbors_2d::t_host_const_randomread t_neighbors_2d_randomread;

};
#endif
//default Gromacs Types
typedef struct ArrayTypes<GMXDeviceType> DAT;
typedef struct ArrayTypes<GMXHostType> HAT;

template<class DeviceType, class BufferView, class DualView>
void buffer_view(BufferView &buf, DualView &view,
                 const size_t n0,
                 const size_t n1 = 0,
                 const size_t n2 = 0,
                 const size_t n3 = 0,
                 const size_t n4 = 0,
                 const size_t n5 = 0,
                 const size_t n6 = 0,
                 const size_t n7 = 0) {

  buf = BufferView(
          view.template view<DeviceType>().ptr_on_device(),
          n0,n1,n2,n3,n4,n5,n6,n7);

}

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


#endif
