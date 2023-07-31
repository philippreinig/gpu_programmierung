#pragma once

#include <cstdlib>
#include <ctime>
#include <iostream>

double rand_double(){
    // srand(time(0));
    return ((double) rand()) / RAND_MAX;
}