
#include <Utilities/Configuration.h>
#include <Utilities/RandomGenerator.h>
#include <QMCWaveFunctions/einspline_spo.hpp>
#include <Numerics/Spline2/MultiBsplineRef.hpp>
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#ifdef USE_GLOBAL_ARRAYS
#include <Utilities/SplineCoefs.hpp>
#endif

using namespace qmcplusplus;

int main(int argc, char **argv)
{
#ifdef HAVE_MPI
#ifdef USE_GLOBAL_ARRAYS
  MPI_Comm m_world = qmcpack::SplineCoefs<OHMMS_PRECISION>::init(argc, argv);
#else
  MPI_Init(&argc, &argv);
  MPI_Comm m_world = MPI_COMM_WORLD;
#endif
#endif
  using spo_type = einspline_spo<OHMMS_PRECISION, MultiBspline<OHMMS_PRECISION>>;
  using spo_ref_type = einspline_spo<OHMMS_PRECISION, miniqmcreference::MultiBsplineRef<OHMMS_PRECISION>>;

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
  std::cout << "evaluate_v from reference implementation" << std::endl;
  std::cout << "  v[0] = " << spo_ref_main.psi[0][0] << std::endl;
  std::cout << "  v[1] = " << spo_ref_main.psi[0][1] << std::endl;

  spo_ref_copy.evaluate_vgh(pos);
  std::cout << "evaluate_vgh from reference implementation" << std::endl;
  std::cout << "  v[0] = " << spo_ref_copy.psi[0][0] << std::endl;
  std::cout << "  v[1] = " << spo_ref_copy.psi[0][1] << std::endl;

  spo_main.evaluate_v(pos);
  std::cout << "evaluate_v from new implementation" << std::endl;
  std::cout << "  v[0] = " << spo_main.psi[0][0] << std::endl;
  std::cout << "  v[1] = " << spo_main.psi[0][1] << std::endl;

  spo_copy.evaluate_vgh(pos);
  std::cout << "evaluate_vgh from new implementation" << std::endl;
  std::cout << "  v[0] = " << spo_copy.psi[0][0] << std::endl;
  std::cout << "  v[1] = " << spo_copy.psi[0][1] << std::endl;


#ifdef HAVE_MPI
#ifdef USE_GLOBAL_ARRAYS
  qmcpack::SplineCoefs<OHMMS_PRECISION>::finalize();
#else
  MPI_Finalize();
#endif
#endif
  return 0;
};
