#include <iostream>
#include <chrono>

#include "../inc/neural_network.hpp"

void print_matrix(Matrix* M){

    for(uint i = 0; i < M->rows; ++i){
        std::cout << "[";
        for(uint j = 0; j < M->cols; ++j){
            if (j < M->cols -1) {
                std::cout << (*M)[i][j] << ", ";
            }
            else{
                std::cout << (*M)[i][j];
            }
        }
        std::cout << "]" << std::endl;

    }

}

double* allocate_matrix(const uint rows, const uint cols){
    return new double[rows*cols];
}



int main (){
    std::vector<unsigned int> topology = {5, 1000, 1000, 1000, 1000, 1000, 5};

    std::vector<double> input_values = {-1,-2,3,-4,5};

    auto neural_network = NeuralNetwork(topology);

    std::cout << "Starting measurement for serial feed forward calculation" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    auto output = neural_network.feed_forward(input_values);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

    std::cout << "Serial feed forward calulation took " << duration << " seconds" << std::endl;
    std::cout << "Result from serial calculation is: " << std::endl;
    output.print();

    // std::cout << "----------------------------------------------------------------" << std::endl;

    // std::cout << "Starting measurement for cache optimized, but serial feed forward calculation" << std::endl;
    // auto start_cache_optimized = std::chrono::high_resolution_clock::now();
    
    // auto output_cache_optimized = neural_network.feed_forward_cache_optimized(input_values);

    // auto end_cache_optimized = std::chrono::high_resolution_clock::now();

    // auto duration_cache_optimized = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

    // std::cout << "Cache optimized feed forward calulation took " << duration_cache_optimized << " seconds" << std::endl;
    // std::cout << "Result from cache optimized feed forward calculation is: " << std::endl;
    // output.print();

    std::cout << "----------------------------------------------------------------" << std::endl;

    auto start_parallel = std::chrono::high_resolution_clock::now();
    
    auto output_parallel = neural_network.feed_forward_parallel(input_values);

    auto end_parallel = std::chrono::high_resolution_clock::now();

    auto duration_parallel = std::chrono::duration_cast<std::chrono::milliseconds>(end_parallel - start_parallel).count() / 1000.0;

    std::cout << "Parallel feed forward calulation took " << duration_parallel << " seconds" << std::endl;
    std::cout << "Result from parallel feed forward calculation is: " << std::endl;
    output_parallel.print();
    
    // neural_network.print();

    return 0;
}
