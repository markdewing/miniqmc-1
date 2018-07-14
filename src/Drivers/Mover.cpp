////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
// Ye Luo, yeluo@anl.gov,
//    Argonne National Laboratory
//
// File created by:
// Ye Luo, yeluo@anl.gov,
//    Argonne National Laboratory
////////////////////////////////////////////////////////////////////////////////


#include <Drivers/Mover.hpp>
#include <Input/Input.hpp>
#include <vector>

namespace qmcplusplus
{
  Mover::Mover(const spo_type &spo_main,
               const int team_size,
               const int member_id,
               const uint32_t myPrime,
               const Tensor<int, 3> &tmat)
    : spo(spo_main, team_size, member_id), rng(myPrime), nlpp(rng)
  {
    ions.setName("ion");
    els.setName("e");

    ions.Lattice.BoxBConds = 1;
    tile_cell(ions, tmat, static_cast<OHMMS_PRECISION>(1.0));
    ions.RSoA = ions.R; // fill the SoA

    const int nions = ions.getTotalNum();
    const int nels  = count_electrons(ions, 1);
    const int nels3 = 3 * nels;

    { // create up/down electrons
      els.Lattice.BoxBConds = 1;
      els.Lattice.set(ions.Lattice);
      std::vector<int> ud(2);
      ud[0] = nels / 2;
      ud[1] = nels - ud[0];
      els.create(ud);
      els.R.InUnit = 1;
      rng.generate_uniform(&els.R[0][0], nels3);
      els.convert2Cart(els.R); // convert to Cartiesian
      els.RSoA = els.R;
    }

  }
}
