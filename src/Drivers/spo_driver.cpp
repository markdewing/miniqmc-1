
#include <Utilities/Configuration.h>
#include <Utilities/RandomGenerator.h>
#include <QMCWaveFunctions/einspline_spo.hpp>
#include <Numerics/Spline2/MultiBsplineRef.hpp>
#ifdef HAVE_MPI
#include <mpi.h>
#endif

using namespace qmcplusplus;

int main(int argc, char **argv)
{
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm m_world = MPI_COMM_WORLD;
#endif
  using spo_type = einspline_spo<double, MultiBspline<double>>;
  using spo_ref_type = einspline_spo<double, miniqmcreference::MultiBsplineRef<double>>;

  spo_type spo_main;
  spo_ref_type spo_ref_main;

  int nx = 5;
  int ny = 5;
  int nz = 5;
  int nspline = 5;
  int ntiles = 1;

  Tensor<double, 3> lattice_b;
  lattice_b(0,0) = 1.0;
  lattice_b(1,1) = 1.0;
  lattice_b(2,2) = 1.0;

  spo_main.set(nx, ny, nz, nspline, ntiles);
  spo_main.Lattice.set(lattice_b);
  spo_ref_main.set(nx, ny, nz, nspline, ntiles);
  spo_ref_main.Lattice.set(lattice_b);

  spo_type spo_copy(spo_main, 1, 0);
  spo_ref_type spo_ref_copy(spo_ref_main, 1, 0);

  typedef TinyVector<double, 3> PosType;
  PosType pos = {0.0, 0.0, 0.0};

  spo_ref_main.evaluate_v(pos);
  std::cout << "ref spo psi[0][0] = " << spo_ref_main.psi[0][0] << std::endl;
  std::cout << "ref spo psi[0][1] = " << spo_ref_main.psi[0][1] << std::endl;

  spo_ref_copy.evaluate_vgh(pos);
  std::cout << "ref spo copy, psi[0][0] = " << spo_ref_copy.psi[0][0] << std::endl;
  std::cout << "ref spo copy, psi[0][1] = " << spo_ref_copy.psi[0][1] << std::endl;

  spo_main.evaluate_v(pos);
  std::cout << "new spo psi[0][0] = " << spo_main.psi[0][0] << std::endl;
  std::cout << "new spo psi[0][1] = " << spo_main.psi[0][1] << std::endl;

  spo_copy.evaluate_vgh(pos);
  std::cout << "copy spo psi[0][0] = " << spo_copy.psi[0][0] << std::endl;
  std::cout << "copy spo psi[0][1] = " << spo_copy.psi[0][1] << std::endl;


#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  return 0;
};
