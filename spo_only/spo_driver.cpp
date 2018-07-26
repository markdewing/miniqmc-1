
#include <vector>
#include <random>
#include <sys/time.h>
#include <stddef.h>

#include <MultiBsplineRef.hpp>
#include <MultiBspline.hpp>
#include <MultiBsplineData.hpp>
#ifdef HAVE_MPI
#include <mpi.h>
#endif

using namespace qmcplusplus;

inline double cpu_clock()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec+(1.e-6)*tv.tv_usec;
}


void setup_spline_data(bspline_traits<double, 3>::SplineType &spline_data, int nspline)
{
  int grid_num = 37;
  int coef_num = grid_num + 3;
  double grid_start = 1.0;
  double grid_end = 1.0;
  spline_data.num_splines = nspline;
  double delta = (grid_start - grid_end)/grid_num;
  double delta_inv = 1.0/delta;

  spline_data.x_grid.start = grid_start;
  spline_data.x_grid.end = grid_end;
  spline_data.x_grid.num = grid_num;
  spline_data.x_grid.delta = delta;
  spline_data.x_grid.delta_inv = delta_inv;
  spline_data.x_stride = coef_num * coef_num * nspline;

  spline_data.y_grid.start = grid_start;
  spline_data.y_grid.end = grid_end;
  spline_data.y_grid.num = grid_num;
  spline_data.y_grid.delta = delta;
  spline_data.y_grid.delta_inv = delta_inv;
  spline_data.y_stride = coef_num * nspline;

  spline_data.z_grid.start = grid_start;
  spline_data.z_grid.end = grid_end;
  spline_data.z_grid.num = grid_num;
  spline_data.z_grid.delta = delta;
  spline_data.z_grid.delta_inv = delta_inv;
  spline_data.z_stride = nspline;

  int coef_size = coef_num * coef_num * coef_num * nspline;
  spline_data.coefs = (double *)malloc(sizeof(double) * coef_size);

  size_t coef_bytes = coef_size * sizeof(double);
  double coef_mb = coef_bytes/(1024.0)/1024; // in MB
  std::cout << " Coefficient size (MB) = " << coef_mb << std::endl;

  std::mt19937 e1(1);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  for (int i = 0; i < coef_size; i++) {
    spline_data.coefs[i] = uniform(e1);
  }
}

int main(int argc, char **argv)
{
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm m_world = MPI_COMM_WORLD;
#endif

  int na = 1;
  int nb = 1;
  int nc = 1;

  //int nx = 5;
  //int ny = 5;
  //int nz = 5;
  int nx = 37;
  int ny = 37;
  int nz = 37;
  //int nspline = 5;
  int nspline = 192;
  int ntiles = 1;


  std::vector<double> psi, psi_ref;
  psi.resize(nspline);
  psi_ref.resize(nspline);

  std::vector<double> grad, grad_ref;
  grad.resize(3*nspline);
  grad_ref.resize(3*nspline);

  std::vector<double> lapl, lapl_ref;
  lapl.resize(3*nspline);
  lapl_ref.resize(3*nspline);

  double pos[3] = {0.0, 0.0, 0.0};
  bspline_traits<double, 3>::SplineType spline_data;

  setup_spline_data(spline_data, nspline);

  miniqmcreference::MultiBsplineRef<double> spline_ref;

  int nrepeat = 100;

  double ref_start = cpu_clock();
  for (int i = 0; i < nrepeat; i++) {
    //spline_ref.evaluate_v(&spline_data, pos[0], pos[1], pos[2], psi_ref.data(), nspline);
    spline_ref.evaluate_vgl(&spline_data, pos[0], pos[1], pos[2], psi_ref.data(), grad_ref.data(), lapl_ref.data(), nspline);
  }

  double ref_end = cpu_clock();

  std::cout << "ref spo psi[0] = " << psi_ref[0] << std::endl;
  std::cout << "ref spo psi[1] = " << psi_ref[1] << std::endl;
  std::cout << " ref time = " << (ref_end - ref_start)/nrepeat << std::endl;


  MultiBspline<double> spline_new;
  double start = cpu_clock();
  for (int i = 0; i < nrepeat; i++) {
    //spline_new.evaluate_v(&spline_data, pos[0], pos[1], pos[2], psi.data(), nspline);
    spline_new.evaluate_vgl(&spline_data, pos[0], pos[1], pos[2], psi.data(), grad.data(), lapl.data(), nspline);
  }
  double end = cpu_clock();

  std::cout << "new spo psi[0] = " << psi[0] << std::endl;
  std::cout << "new spo psi[1] = " << psi[1] << std::endl;
  std::cout << " time = " << (end - start)/nrepeat << std::endl;


#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  return 0;
};
