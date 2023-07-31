#ifndef _A1_HPP_
#define _A1_HPP_

// Under NO circumstance are you to change the given function signatures (it would break the automatic grading).
// You can add functions if you want but do NOT change the signature of the given functions.

// Known problem: The worker functions are overloaded so std::thread can not choose which to use.
// Solution: create a function pointer with the required signature explicitly
// RETURN_VALUE (*POINTER_NAME)(PARAMETER_TYPE_1, PARAMETER_TYPE_2, ...) = MatrixMultiplicationWorker;
// worker = std::thread(POINTER_NAME, ...);

/** \brief The namespace of the first assignment
*/
namespace a1{
  /**\brief A serial implementation of a matrix-matrix multiplication C=A*B for square matrices.
  *
  * This function does not reserve any memory on the heap. Thus C is managed by the calling function.
  * However make sure that C is overriden and contains the right result.
  *
  * @param A a row-major order matrix that is the left hand side of the multiplication.
  * @param B a row-major order matrix that is the right hand side of the multiplication
  * @param C a row-major order matrix that is the result of the multiplication
  * @param size the size of one dimension for the square matrices.
  */
  void MatrixMultiplicationSerial(const double* A, const double* B, double* C, const unsigned int size);

  /**\brief A worker function for the parallel matrix-matrix multiplication.
  *
  * This function does not reserve any memory on the heap. Thus C is managed by the calling function.
  * This function does not use any synchronization primitives.
  *
  * @param A a row-major order matrix that is the left hand side of the multiplication.
  * @param B a row-major order matrix that is the right hand side of the multiplication
  * @param C a row-major order matrix that is the result of the multiplication
  * @param size the size of one dimension for the square matrices.
  * @param tid the thread id
  * @param num_threads the number of threads working on this multiplication.
  */
  void MatrixMultiplicationWorker(const double* A, const double* B, double* C, const unsigned int size, const unsigned int tid, const unsigned int num_threads);

  /**\brief A parallel implementation of a matrix-matrix multiplication C=A*B for square matrices.
  *
  * This function does not reserve any memory on the heap. Thus C is managed by the calling function.
  *
  * @param A a row-major order matrix that is the left hand side of the multiplication.
  * @param B a row-major order matrix that is the right hand side of the multiplication
  * @param C a row-major order matrix that is the result of the multiplication
  * @param size the size of one dimension for the square matrices.
  * @param num_threads the number of threads working on this multiplication.
  */
  void MatrixMultiplicationParallel(const double* A, const double* B, double* C, const unsigned int size, const unsigned int num_threads);

////////////////////////////////////////////////////////////////////////////////
/// BONUS
////////////////////////////////////////////////////////////////////////////////

  /**\brief A serial implementation of a matrix-matrix multiplication C=A*B.
  *
  * This function does not reserve any memory on the heap. Thus C is managed by the calling function.
  *
  * @param A a row-major order matrix of dimensions N*K (N rows, K cols)
  * @param B a row-major order matrix of dimensions K*M (K rows, M cols)
  * @param C the resulting row-major order matrix of dimensions N*M (N rows, M cols)
  * @param N number of rows of A and C
  * @param K number of cols of A and rows of B
  * @param M number of cols of B and C
  */
  void MatrixMultiplicationSerial(const double* A, const double* B, double* C, const unsigned int N, const unsigned int K, const unsigned int M);

  /**\brief A worker function for the parallel matrix-matrix multiplication.
  *
  * This function does not reserve any memory on the heap. Thus C is managed by the calling function.
  * This function does not use any synchronization primitives.
  *
  * @param A a row-major order matrix of dimensions N*K (N rows, K cols)
  * @param B a row-major order matrix of dimensions K*M (K rows, M cols)
  * @param C the resulting row-major order matrix of dimensions N*M (N rows, M cols)
  * @param N number of rows of A and C
  * @param K number of cols of A and rows of B
  * @param M number of cols of B and C
  * @param tid the thread id
  * @param num_threads the number of threads working on this multiplication.
  */
  void MatrixMultiplicationWorker(const double* A, const double* B, double* C, const unsigned int N, const unsigned int K, const unsigned int M, const unsigned int tid, const unsigned int num_threads);

  /**\brief A parallel implementation of a matrix-matrix multiplication C=A*B.
  *
  * This function does not reserve any memory on the heap. Thus C is managed by the calling function.
  *
  * @param A a row-major order matrix of dimensions N*K (N rows, K cols)
  * @param B a row-major order matrix of dimensions K*M (K rows, M cols)
  * @param C the resulting row-major order matrix of dimensions N*M (N rows, M cols)
  * @param N number of rows of A and C
  * @param K number of cols of A and rows of B
  * @param M number of cols of B and C
  * @param num_threads the number of threads working on this multiplication.
  */
  void MatrixMultiplicationParallel(const double* A, const double* B, double* C, const unsigned int N, const unsigned int K, const unsigned int M, const unsigned int num_threads);

}


#endif /* end of include guard: _A1_HPP_ */
