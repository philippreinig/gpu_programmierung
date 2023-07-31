#include <a2.hpp>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <cmath>

void fill_array_with_rndm_ints(int* array, int size){
  
  srand(time(NULL));

  for(int i = 0; i < size; ++i){
    array[i] = rand() % 10 + 1;
  }

}

void fill_array_ascending(int* array, int size){
  for(int i = 0; i < size; ++i){
    array[i] = i;
  }
}

void fill_array_with_specific_value(int* array, int size, int value){
  std::fill_n(array, size, value);
}

int main(int, char**){
  for(int i = 10; i < 24; ++i){
    int size = std::pow(2, i);
    printf("i: %d, size: %d\n", i, size);
    int* array = new int[size];
    int reference_result;
    int parallel_result;

  // int** array_device;

  // a2::initDeviceMemory(array, array_device, size);

  // fill_array_ascending(array, size);
  fill_array_with_specific_value(array, size, 1);
  
  std::cout << "Array creation done" << std::endl;

  float reference_calculation_duration = a2::reference(array, size, reference_result);

  std::cout << "Reference result: " << reference_result << std::endl;
  std::cout << "Reference calculation duration: " << reference_calculation_duration << "s" << std::endl;

  float parallel_calculation_duration = a2::version1(array, size, parallel_result);

  std::cout << "Parallel result: " << parallel_result << std::endl;

  std::cout << "Parallel calculation duration: " << parallel_calculation_duration << "s" << std::endl;

  free(array);
  }
  return 0;
}
