#include <a1.hpp>
#include <iostream>
#include <thread>
#include <vector>

/** \brief The namespace of the first assignment
*/
namespace a1{
  /**\brief A serial implementation of a matrix-matrix multiplication C=A*B for square matrices.
  *
  * This function does not reserve any memory on the heap. Thus C is managed by the calling function.
  *
  * @param A a row-major order matrix that is the left hand side of the multiplication.
  * @param B a row-major order matrix that is the right hand side of the multiplication
  * @param C a row-major order matrix that is the result of the multiplication
  * @param size the size of one dimension for the square matrices.
  */
  void MatrixMultiplicationSerial(const double* A, const double* B, double* C, const unsigned int size){
    for(unsigned int i = 0; i < size; ++i){
        for(unsigned int j = 0; j < size; ++j){
            double sum = 0;
            for(unsigned int k = 0; k < size; ++k){
                sum += A[i*size+k] * B[k*size+j];
            }
            C[i*size+j] = sum;
        }
    }

  }

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
  void MatrixMultiplicationWorker(const double* A, const double* B, double* const C, const unsigned int size, const unsigned int tid, const unsigned int num_threads){
      for(unsigned int i = tid; i < size; i += num_threads){
          // std::cout << "Thread " << tid << " calculating line " << i << std::endl;
          for(unsigned int j = 0; j < size; ++j){
              double sum = 0;
              for(unsigned int k = 0; k < size; ++k){
                  sum += A[i*size+k] * B[k*size+j];
              }
              C[i*size+j] = sum;
          }
      }
  }




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
  void MatrixMultiplicationParallel(const double* const A, const double* const B, double* const C, const unsigned int size, unsigned int num_threads){
      if (num_threads > size){
          std::cout << "Requested amount of threads (" << num_threads << ") is greater than the size of the matrix (" << size <<") -> Restricting thread amount to " << size << std::endl;
          num_threads = size;
      }

      std::vector<std::thread> threads;
      void (*matrix_multiplication_parallel_worker)(const double* A, const double* B, double* const C, const unsigned int size, const unsigned int tid, const unsigned int num_threads) = MatrixMultiplicationWorker;
      for (unsigned int i = 0; i < num_threads; i++) {
          threads.emplace_back(matrix_multiplication_parallel_worker, A, B, C, size, i, num_threads);
      }

      for(auto& thread : threads){
          thread.join();
      }
  }

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
  void MatrixMultiplicationSerial(const double* A, const double* B, double* C, const unsigned int N, const unsigned int K, const unsigned int M){

  }

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
  void MatrixMultiplicationWorker(const double* A, const double* B, double* C, const unsigned int N, const unsigned int K, const unsigned int M, const unsigned int tid, const unsigned int num_threads){

  }

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
  void MatrixMultiplicationParallel(const double* A, const double* B, double* C, const unsigned int N, const unsigned int K, const unsigned int M, const unsigned int num_threads){

  }

}
