//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2018 Jeongnim Kim and QMCPACK developers.
//
// File developed by:  Mark Dewing, mdewing@anl.gov Argonne National Laboratory
//
// File created by: Mark Dewing, mdewing@anl.gov Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


/** @file Communicate.cpp
 * @brief Defintion of Communicate and CommunicateMPI classes.
 */
#include <Utilities/Communicate.h>
#include <Utilities/Configuration.h>
#include <Utilities/SplineCoefs.hpp>
#include <iostream>

Communicate::Communicate(int argc, char **argv)
{
#ifdef HAVE_MPI
#ifdef USE_GLOBAL_ARRAYS
  m_world = qmcpack::SplineCoefs<qmcplusplus::QMCTraits::RealType>::init(argc, argv);
  MPI_Comm_rank(m_world, &m_rank);
  MPI_Comm_size(m_world, &m_size);
#else
  MPI_Init(&argc, &argv);
  m_world = MPI_COMM_WORLD;
  MPI_Comm_rank(m_world, &m_rank);
  MPI_Comm_size(m_world, &m_size);
#endif
#else
  m_rank = 0;
  m_size = 1;
#endif
}

Communicate::~Communicate()
{
#ifdef HAVE_MPI
#ifdef USE_GLOBAL_ARRAYS
  qmcpack::SplineCoefs<qmcplusplus::QMCTraits::RealType>::finalize();
#else
  MPI_Finalize();
#endif
#endif
}

void
Communicate::reduce(int &value)
{
#ifdef HAVE_MPI
  int local_value = value;
  MPI_Reduce(&local_value, &value, 1, MPI_INT, MPI_SUM, 0, m_world);
#endif
}
