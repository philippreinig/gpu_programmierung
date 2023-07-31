#pragma once

#include <vector>

#include "neuron.hpp"
#include "matrix.hpp"

class Layer{
    public:
        Layer(unsigned int indx, uint size){
            neurons = std::vector<Neuron>(size);
            this->indx = indx;
        }

        uint get_size(){
            return this->neurons.size();
        }

        Neuron& get_neuron(uint indx){
            return neurons[indx];
        }

        Matrix* get_value_matrix(){
            auto m = new Matrix(1, this->neurons.size());
            for(uint i = 0; i < neurons.size(); ++i){
                (*m)[0][i] = neurons[i].get_value();
            }
            return m;
        }

        void set_values(std::vector<double> values){
            if(this->neurons.size() != values.size()){
                std::cout << "Layer::set_neurons(std::vector<double>): Layer " << this->indx << ": Sizes don't match: Should be " << this->neurons.size() << ", but is: " << neurons.size() << std::endl;
                throw std::invalid_argument("Sizes don't match");
            }

            for(uint i = 0; i < neurons.size(); ++i){
                this->neurons[i].set_value(values[i]);
            }
        }

        void set_values(double* values){
            for(uint i = 0; i < neurons.size(); ++i){
                this->neurons[i].set_value(values[i]);
            }
        }

    private: 
        std::vector<Neuron> neurons;
        unsigned int indx;
};