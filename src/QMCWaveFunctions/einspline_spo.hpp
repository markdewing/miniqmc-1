////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
// Amrita Mathuriya, amrita.mathuriya@intel.com,
//    Intel Corp.
//
// File created by:
// Jeongnim Kim, jeongnim.kim@intel.com,
//    Intel Corp.
// ////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
/** @file einspline_spo.hpp
 */
#ifndef QMCPLUSPLUS_EINSPLINE_SPO_HPP
#define QMCPLUSPLUS_EINSPLINE_SPO_HPP
#include <Utilities/Configuration.h>
#include <Particle/ParticleSet.h>
#include <Numerics/Spline2/bspline_allocator.hpp>
#include <Numerics/Spline2/MultiBspline.hpp>
#include <Utilities/SIMD/allocator.hpp>
#include "Numerics/OhmmsPETE/OhmmsArray.h"
#include <iostream>

namespace qmcplusplus
{
template <typename T, typename compute_engine_type = MultiBspline<T> >
struct einspline_spo
{
  /// define the einsplie data object type
  using spline_type = typename bspline_traits<T, 3>::SplineType;
  using pos_type        = TinyVector<T, 3>;
  using vContainer_type = aligned_vector<T>;
  using gContainer_type = VectorSoAContainer<T, 3>;
  using hContainer_type = VectorSoAContainer<T, 6>;
  using lattice_type    = CrystalLattice<T, 3>;

  /// number of blocks
  int nBlocks;
  /// first logical block index
  int firstBlock;
  /// last gical block index
  int lastBlock;
  /// number of splines
  int nSplines;
  /// number of splines per block
  int nSplinesPerBlock;
  /// if true, responsible for cleaning up einsplines
  bool Owner;
  lattice_type Lattice;
  /// use allocator
  einspline::Allocator myAllocator;
  /// compute engine
  compute_engine_type compute_engine;

  aligned_vector<spline_type *> einsplines;
  aligned_vector<vContainer_type> psi;
  aligned_vector<gContainer_type> grad;
  aligned_vector<hContainer_type> hess;

  /// default constructor
  einspline_spo()
      : nBlocks(0), nSplines(0), firstBlock(0), lastBlock(0), Owner(false)
  {
  }
  /// disable copy constructor
  einspline_spo(const einspline_spo &in) = delete;
  /// disable copy operator
  einspline_spo &operator=(const einspline_spo &in) = delete;

  /** copy constructor
   * @param in einspline_spo
   * @param team_size number of members in a team
   * @param member_id id of this member in a team
   *
   * Create a view of the big object. A simple blocking & padding  method.
   */
  einspline_spo(einspline_spo &in, int team_size, int member_id)
      : Owner(false), Lattice(in.Lattice)
  {
    nSplines         = in.nSplines;
    nSplinesPerBlock = in.nSplinesPerBlock;
    nBlocks          = (in.nBlocks + team_size - 1) / team_size;
    firstBlock       = nBlocks * member_id;
    lastBlock        = std::min(in.nBlocks, nBlocks * (member_id + 1));
    nBlocks          = lastBlock - firstBlock;
    einsplines.resize(nBlocks);
    for (int i = 0, t = firstBlock; i < nBlocks; ++i, ++t)
      einsplines[i] = in.einsplines[t];
    resize();
  }

  /// destructors
  ~einspline_spo()
  {
    if (Owner)
      for (int i = 0; i < nBlocks; ++i)
        myAllocator.destroy(einsplines[i]);
  }

  /// resize the containers
  void resize()
  {
    psi.resize(nBlocks);
    grad.resize(nBlocks);
    hess.resize(nBlocks);
    for (int i = 0; i < nBlocks; ++i)
    {
      psi[i].resize(nSplinesPerBlock);
      grad[i].resize(nSplinesPerBlock);
      hess[i].resize(nSplinesPerBlock);
    }
  }

  // fix for general num_splines
  void set(int nx, int ny, int nz, int num_splines, int nblocks,
           bool set_local_coefs,
           bool init_random = true)
  {
    nSplines         = num_splines;
    nBlocks          = nblocks;
    nSplinesPerBlock = num_splines / nblocks;
    firstBlock       = 0;
    lastBlock        = nBlocks;
    if (einsplines.empty())
    {
      //global_coefs = new qmcpack::SplineCoefs<T>(nx+3, ny+3, nz+3, num_splines);
      //coefs_local_buf = new T[4*(nz+3)*num_splines];
      Owner = true;
      TinyVector<int, 3> ng(nx, ny, nz);
      pos_type start(0);
      pos_type end(1);
      einsplines.resize(nBlocks);
      RandomGenerator<T> myrandom(11);
      Array<T, 3> coef_data(nx+3, ny+3, nz+3);
#ifdef USE_GLOBAL_ARRAYS
      bool alloc_coefs = set_local_coefs;
#else
      bool alloc_coefs = true;
#endif
      for (int i = 0; i < nBlocks; ++i)
      {
        einsplines[i] = myAllocator.createMultiBspline(T(0), start, end, ng, PERIODIC, nSplinesPerBlock, alloc_coefs);
#ifdef USE_GLOBAL_ARRAYS
#ifdef USE_Z_PADDING
        intptr_t zs = einsplines[i]->z_stride;
#else
        intptr_t zs = num_splines;
#endif
        global_coefs = new qmcpack::SplineCoefs<T>(nx+3, ny+3, nz+3, zs);
        coefs_local_buf = new T[4*(nz+3)*zs];
        CoefFetchTimer = TimerManager.createTimer("Coef Fetch", timer_level_fine);
#endif
        if (init_random) {
          for (int j = 0; j < nSplinesPerBlock; ++j) {
            // Generate different coefficients for each orbital
            myrandom.generate_uniform(coef_data.data(), coef_data.size());
#ifdef USE_GLOBAL_ARRAYS
            global_coefs->writeCoefs(j, coef_data.data());
            // also include this line when testing (running check_spo)
            if (set_local_coefs){
              myAllocator.setCoefficientsForOneOrbital(j, coef_data, einsplines[i]);
            }
#else
            myAllocator.setCoefficientsForOneOrbital(j, coef_data, einsplines[i]);
#endif
          }
        }
      }
      global_coefs->set();
    }
    resize();
  }

  /** evaluate psi */
  inline void evaluate_v(const pos_type &p)
  {
    auto u = Lattice.toUnit_floor(p);
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_v(einsplines[i], u[0], u[1], u[2], psi[i].data(), nSplinesPerBlock);
  }

  /** evaluate psi */
  inline void evaluate_v_pfor(const pos_type &p)
  {
    auto u = Lattice.toUnit_floor(p);
    #pragma omp for nowait
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_v(einsplines[i], u[0], u[1], u[2], psi[i].data(), nSplinesPerBlock);
  }

  /** evaluate psi, grad and lap */
  inline void evaluate_vgl(const pos_type &p)
  {
    auto u = Lattice.toUnit_floor(p);
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgl(einsplines[i], u[0], u[1], u[2],
                                  psi[i].data(), grad[i].data(), hess[i].data(),
                                  nSplinesPerBlock);
  }

  /** evaluate psi, grad and lap */
  inline void evaluate_vgl_pfor(const pos_type &p)
  {
    auto u = Lattice.toUnit_floor(p);
    #pragma omp for nowait
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgl(einsplines[i], u[0], u[1], u[2],
                                  psi[i].data(), grad[i].data(), hess[i].data(),
                                  nSplinesPerBlock);
  }

  /** evaluate psi, grad and hess */
  inline void evaluate_vgh(const pos_type &p)
  {
    auto u = Lattice.toUnit_floor(p);
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgh(einsplines[i], u[0], u[1], u[2],
                                  psi[i].data(), grad[i].data(), hess[i].data(),
                                  nSplinesPerBlock);
  }

  /** evaluate psi, grad and hess */
  inline void evaluate_vgh_pfor(const pos_type &p)
  {
    auto u = Lattice.toUnit_floor(p);
    #pragma omp for nowait
    for (int i = 0; i < nBlocks; ++i)
      compute_engine.evaluate_vgh(einsplines[i], u[0], u[1], u[2],
                                  psi[i].data(), grad[i].data(), hess[i].data(),
                                  nSplinesPerBlock);
  }

  void print(std::ostream &os)
  {
    os << "SPO nBlocks=" << nBlocks << " firstBlock=" << firstBlock
       << " lastBlock=" << lastBlock << " nSplines=" << nSplines
       << " nSplinesPerBlock=" << nSplinesPerBlock << std::endl;
  }
};
}

#endif
