#pragma once

#include <cmath>

class Neuron{
    public:
        Neuron() : value(0.0){}

        Neuron(double value): value(value){}

        double get_value(){
            return this->value;
        }

        void set_value(double value){
            this->value = 1 / (1 + std::pow(std::exp(1.0), value)); // Threshold activation function
        }

        void init(double value){
            this->value = value;
        }

    private:
        double value;
};