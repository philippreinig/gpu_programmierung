#include <a1.hpp>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <cmath>

void print_matrix(double* M, const unsigned int SIZE){

    for(unsigned int i = 0; i < SIZE; ++i){
        std::cout << "[";
        for(unsigned int j = 0; j < SIZE; ++j){
            if (j < SIZE -1) {
                std::cout << M[i*SIZE+j] << ", ";
            }
            else{
                std::cout << M[i*SIZE+j];
            }
        }
        std::cout << "]" << std::endl;

    }

}

double* create_random_square_matrix(const unsigned int size){
    double* const M = new double[size*size];

    for(unsigned int i = 0; i < size*size; ++i){
        M[i] = rand() % 10;
    }

    return M;

}

int* matrix_equality_check(const double* A, const double* B, const int size){
    for(int i = 0; i < size; ++i){
        for(int j = 0; j < size; ++j){
            if(A[i*size+j] != B[i*size+j]) return new int[2]{i,j};
        }
    }
    return nullptr;
}

int main(int, char**) {
    for (unsigned int i = 6; i <= 11; ++i) {


        const int SIZE = pow(2, i);


        double *A = create_random_square_matrix(SIZE);

        double *B = create_random_square_matrix(SIZE);

        double *C_serial = new double[SIZE * SIZE];

        double *C_parallel = new double[SIZE * SIZE];

        //std::cout << "=== A ===" << std::endl;
        //print_matrix(A, SIZE);
        //std::cout << "=== B ===" << std::endl;
        //print_matrix(B, SIZE);

        // -------------------------- Serial Matrix Multiplication --------------------------

        auto start_serial = std::chrono::high_resolution_clock::now();

        a1::MatrixMultiplicationSerial(A, B, C_serial, SIZE);

        auto end_serial = std::chrono::high_resolution_clock::now();

        auto serial_execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_serial - start_serial);

        //std::cout << "=== C_serial ===" << std::endl;
        //print_matrix(C_serial, SIZE);

        std::cout << "Serial multiplication of matrix with size " << SIZE << " took " << serial_execution_time.count() << "ms." << std::endl;


        // -------------------------- Parallel Matrix Multiplication --------------------------

//    int thread_amounts[] = {2,3,4,5,6,7,8,9,10,12,15,20,30,40,50};


        int num_threads = 5;

//    for(int num_threads : thread_amounts) {

        auto start_parallel = std::chrono::high_resolution_clock::now();

        a1::MatrixMultiplicationParallel(A, B, C_parallel, SIZE, num_threads);

        auto end_parallel = std::chrono::high_resolution_clock::now();

        auto parallel_execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_parallel - start_parallel);

        //std::cout << "=== C_parallel ===" << std::endl;
        //print_matrix(C_parallel, SIZE);

        std::cout << "Parallel multiplication of matrix with size " << SIZE  << " and " << num_threads << " threads took "
                  << parallel_execution_time.count() << "ms." << std::endl;

//        int *matrices_difference_indices = matrix_equality_check(C_serial, C_parallel, SIZE);
//
//        if (matrices_difference_indices == nullptr) {
//            std::cout << "Serial and parallel matrix multiplication results are equal" << std::endl;
//        } else {
//            std::cout << "else" << std::endl;
//            std::cout << "Serial and parallel matrix multiplication results differ at "
//                      << matrices_difference_indices[0] << ", " << matrices_difference_indices[1]
//                      << ": " << C_serial[matrices_difference_indices[0] * SIZE + matrices_difference_indices[1]]
//                      << " vs. "
//                      << C_parallel[matrices_difference_indices[0] * SIZE + matrices_difference_indices[1]]
//                      << std::endl;
//            free(matrices_difference_indices);
//        }



    // // -------------------------- Free Allocated Memory --------------------------

    free(A);
    free(B);
    free(C_serial);
    free(C_parallel);

    }

  return 0;
}
