#ifndef _spline_coefs_h
#define _spline_coefs_h

#include <complex>
#include <typeinfo>
#include <ga.h>
#include <ga-mpi.h>

namespace qmcpack{

// Outline of a simple class for creating a distributed table of spline
// coefficients. The list is assumed to be of size IxJxKxN and the
// indices in all functions are global in the sense that they do not
// depend on how the table is partitioned across processors.
template <class T>
class SplineCoefs {

  public:

  // Initialize Global Arrays library used by SplineCoefs class. Return
  // an MPI communicator that can be used as the "world" communicator by
  // the rest of the application
  static MPI_Comm init(int argc, char **argv)
  {
    MPI_Init(&argc, &argv);
    GA_Initialize();
    return GA_MPI_Comm();
  }

  // Clean up the Global Arrays library at the end of the calculation.
  static void finalize()
  {
    GA_Terminate();
    MPI_Finalize();
  }

  // Constructor for a spline coefficient object of size xdim X ydim X zdim X
  // ndim. This creates a distributed array on the world group (communicator)
  SplineCoefs(int xdim, int ydim, int zdim, int ndim)
  {
    p_GAgrp = GA_Pgroup_get_world();

    // Find data type
    if (typeid(T) == typeid(double)) {
      p_type = C_DBL;
    } else if (typeid(T) == typeid(float)) {
      p_type = C_FLOAT;
    } else if (typeid(T) == typeid(std::complex<double>)) {
      p_type = C_DCPL;
    } else if (typeid(T) == typeid(std::complex<float>)) {
      p_type = C_SCPL;
    }

    int N = 4;
    int dims[4];

    //create dims array
    dims[0] = xdim;
    dims[1] = ydim;
    dims[2] = zdim;
    dims[3] = ndim;

    p_xdim = xdim;
    p_ydim = ydim;
    p_zdim = zdim;
    p_ndim = ndim;

    // create GA to hold coefficients
    int chunk[4];
    if (xdim*ydim >= GA_Nnodes()) {
      // Do not split across last two dimensions
      chunk[0] = -1;
      chunk[1] = -1;
      chunk[2] = zdim;
      chunk[3] = ndim;
    } else {
      // Do not split across last dimension
      chunk[0] = -1;
      chunk[1] = -1;
      chunk[2] = -1;
      chunk[3] = ndim;
    }
    p_GA = GA_Create_handle();
    GA_Set_data(p_GA,N,dims,p_type);
    GA_Set_chunk(p_GA, chunk);
    if (!GA_Allocate(p_GA)) {
      GA_Error("Failure to allocate SplineCoefs array",0);
    }
  }

  // Constructor for a spline coefficient object of size xdim X ydim X zdim X
  // ndim on a subset of processes. The number of processors in this subset is
  // nproc and the processor ranks are given by the list array.
  SplineCoefs(int xdim, int ydim, int zdim, int ndim, int nproc, int *list)
  {
    // Create process group containing only the processors in list
    p_GAgrp = GA_Pgroup_create(list, nproc);

    int N = 4;
    int dims[4];

    // Find data type
    if (typeid(T) == typeid(double)) {
      p_type = C_DBL;
    } else if (typeid(T) == typeid(float)) {
      p_type = C_FLOAT;
    } else if (typeid(T) == typeid(std::complex<double>)) {
      p_type = C_DCPL;
    } else if (typeid(T) == typeid(std::complex<float>)) {
      p_type = C_SCPL;
    }

    //create dims array
    dims[0] = xdim;
    dims[1] = ydim;
    dims[2] = zdim;
    dims[3] = ndim;

    p_xdim = xdim;
    p_ydim = ydim;
    p_zdim = zdim;
    p_ndim = ndim;

    // create GA to hold coefficients
    int chunk[4];
    if (xdim*ydim >= nproc) {
      // Do not split across last two dimensions
      chunk[0] = -1;
      chunk[1] = -1;
      chunk[2] = zdim;
      chunk[3] = ndim;
    } else {
      // Do not split across last dimension
      chunk[0] = -1;
      chunk[1] = -1;
      chunk[2] = -1;
      chunk[3] = ndim;
    }
    p_GA = GA_Create_handle();
    GA_Set_data(p_GA,N,dims,p_type);
    GA_Set_pgroup(p_GA,p_GAgrp);
    GA_Set_chunk(p_GA, chunk);
    if (!GA_Allocate(p_GA)) {
      GA_Error("Failure to allocate SplineCoefs array",0);
    }
  }

  // Simple destructor for the SplineCoefs object
  ~SplineCoefs()
  {
    GA_Destroy(p_GA);
    if (p_GAgrp != GA_Pgroup_get_world()) {
      GA_Pgroup_destroy(p_GAgrp);
    }
  }

  // Write coefficients from local buffer to SplineCoefs object. This function
  // assumes that we do this one block at a time. All spatial indices are
  // included in each block. The index n identifies which block of data is being
  // stored and ptr is the address of the local memory buffer containing the
  // coefficients that are to be copied into the distributed SplineCoefs object.
  // It is also possible to write larger blocks of coefficients at once, if
  // desirable.
  void writeCoefs(int n, T *ptr)
  {
    int lo[4], hi[4], ld[3];
    lo[0] = 0;
    lo[1] = 0;
    lo[2] = 0;
    lo[3] = n;
    hi[0] = p_xdim-1;
    hi[1] = p_ydim-1;
    hi[2] = p_zdim-1;
    hi[3] = n;
    ld[0] = p_ydim;
    ld[1] = p_ydim;
    ld[2] = 1;
    NGA_Put(p_GA,lo,hi,ptr,ld);
  }

  // Call this function after writing all values to the SplineCoefs object and
  // before making any calls to readCoefs. This is collective on all processors
  // and guarantees that the SplineCoefs object is in a known state. Also call
  // this function between epochs that are using GA and epochs that are using
  // MPI to make sure that MPI calls support GA do not get confused with other
  // MPI calls.
  void set()
  {
    GA_Pgroup_sync(p_GAgrp);
  }

  // Read coeficients from SplineCoefs object and copy them to a local buffer.
  // The indices identify the requested block of coefficients and ptr is the
  // address of a local buffer into which the data is copied. If the size of
  // the data request varies from one call to another, we can add more parameters
  // to the argument list.
  void readCoefs(int ix, int iy, int iz, T *ptr)
  {
    int lo[4], hi[4], ld[3];
    lo[0] = ix;
    lo[1] = iy;
    lo[2] = iz;
    lo[3] = 0;
    hi[0] = ix;
    hi[1] = iy;
    hi[2] = iz;
    hi[3] = p_ndim-1;
    ld[0] = 1;
    ld[1] = 1;
    ld[2] = p_ndim;
    NGA_Get(p_GA,lo,hi,ptr,ld);
  }

  // Read a slice of data for a range of values along the z dimension into a
  // local buffer for a given set of x and y indices. The indices identify the x
  // and y coordinates of the slice, izlo and izhi are a range of z-indices that
  // should be copied into the buffer, and T is a pointer to a local buffer.
  void readZSlice(int ix, int iy, int izlo, int izhi, T *ptr)
  {
    int lo[4], hi[4], ld[3];
    lo[0] = ix;
    lo[1] = iy;
    lo[2] = izlo;
    lo[3] = 0;
    hi[0] = ix;
    hi[1] = iy;
    hi[2] = izhi;
    hi[3] = p_ndim-1;
    ld[0] = 1;
    ld[1] = izhi-izlo+1;
    ld[2] = p_ndim;
    NGA_Get(p_GA,lo,hi,ptr,ld);
  }

  // Read an entire slice of data for all values along the z dimension into a
  // local buffer for a given set of x and y indices. The indices identify the x
  // and y coordinates of the slice and T is a pointer to a local buffer.
  void readZSlice(int ix, int iy, T *ptr)
  {
    int lo[4], hi[4], ld[3];
    lo[0] = ix;
    lo[1] = iy;
    lo[2] = 0;
    lo[3] = 0;
    hi[0] = ix;
    hi[1] = iy;
    hi[2] = p_zdim-1;
    hi[3] = p_ndim-1;
    ld[0] = 1;
    ld[1] = p_zdim;
    ld[2] = p_ndim;
    NGA_Get(p_GA,lo,hi,ptr,ld);
  }

  private:

  // Internal GA group handle representing set of processors over which
  // SplineCoefs object is defined
  int p_GAgrp;

  // Internal GA handle for distributed array
  int p_GA;

  // Internal dimensions of distributed coefficient array
  int p_xdim, p_ydim, p_zdim, p_ndim;

  // Coefficient data type
  int p_type;
};

}
#endif  // _spline_coefs_h
