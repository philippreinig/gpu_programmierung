#pragma once

#include <vector>


#include "matrix.hpp"
#include "layer.hpp"

// Amount of weight matrices = Amount of layers - 1;

class NeuralNetwork{
    public:

        NeuralNetwork(std::vector<unsigned int> topology){
            
            if (topology.size() < 2){
                throw std::invalid_argument("Toplogy description has to contain at least 2 layers");
            }

            for(uint i = 0; i < topology.size(); ++i){
                auto layer = Layer(i, topology[i]);
                this->layers.emplace_back(layer);
            }
            std::cout << "Layers intialized" << std::endl;

            for(uint i = 0; i < topology.size() - 1; ++i){
                auto weight_matrix = Matrix(topology[i], topology[i+1]);

                weight_matrix.fill_randomly();

                this->weights.emplace_back(weight_matrix);
            }

            std::cout << "Random weight matrices initialized" << std::endl;
            std::cout << "ANN intialized successfully" << std::endl;
        }

        Matrix feed_forward_parallel(std::vector<double> input_values){
            this->init(input_values);

            auto last_layer_size = this->layers[this->layers.size()-1].get_size();
            Matrix result(1, last_layer_size);
            
            for(uint i = 0; i < this->layers.size() - 1; ++i){
                uint m_vals_rows = 1;
                uint m_vals_cols = this->layers[i].get_size();

                uint m_weights_rows = this->layers[i].get_size();
                uint m_weights_cols = this->layers[i+1].get_size();

                uint m_res_rows = m_vals_rows;
                uint m_res_cols = m_weights_cols;

                auto values = this->layers[i].get_value_matrix()->get_values();
                auto weights = this->weights[i].get_values();
                auto res = (double*) calloc(m_res_cols * m_res_rows, sizeof(double));

                // std::cout << "Feed forward iteration: " << i << std::endl;
                // std::cout << " --- M_1 --- " << std::endl;
                // m_1->print();
                // std::cout << " --- M_2 --- " << std::endl;
                // // m_2.print();

                multiply_parallel(values, m_vals_rows, m_vals_cols,
                                            weights, m_weights_rows, m_weights_cols,
                                            res);

                // std::cout << " --- M_3 --- " << std::endl;
                // m_3->print();
                
                this->layers[i+1].set_values(res);

                // If it's the last layer copy the calculated values to the result matrix which will be returned by this function
                if(i == this->layers.size()-2){
                    if(m_res_cols != result.cols){
                        throw "Implementation error!";
                    }

                    for(uint a = 0; a < last_layer_size; ++a){
                        result[0][a] = res[a];
                    }
                }
                
                delete values;
                delete res;
            }

            return result;
        }

        Matrix feed_forward_cache_optimized(std::vector<double> input_values){
            this->init(input_values);

            auto last_layer_size = this->layers[this->layers.size()-1].get_size();
            Matrix result(1, last_layer_size);

             for(uint i = 0; i < this->layers.size() -1; ++i){
                auto m_1 = this->layers[i].get_value_matrix();
                auto m_2 = this->weights[i];
                auto m_3 = new Matrix(m_1->rows, m_2.cols);

                // std::cout << "Feed forward iteration: " << i << std::endl;
                // std::cout << " --- M_1 --- " << std::endl;
                // // m_1->print();
                // std::cout << " --- M_2 --- " << std::endl;
                // m_2.print();

                Matrix::multiply_cache_optimized(m_1, &m_2, m_3);

                // std::cout << " --- M_3 --- " << std::endl;
                // m_3->print();

                for(uint j = 0; j < m_3->cols; ++j){
                    // auto layer = this->layers[i+1];
                    // auto val = (*m_3)[0][j];
                    // std::cout<<"net[" << i+1 << "][" << j << "]" << val << std::endl;
                    // layer.get_neuron(j).set_value(val);
                    auto vals = std::vector<double>(m_3->cols);
                    for(uint i = 0; i < m_3->cols; ++i){
                        vals[i] = (*m_3)[0][i];
                    }
                    this->layers[i+1].set_values(vals);
                }
                
                
                
                // If it's the last layer copy the calculated values to the matrix result
                if(i == this->layers.size()-2){
                    if(m_3->cols != result.cols){
                        throw "Implementation error!";
                    }
                    for(uint a = 0; a < last_layer_size; ++a){
                        result[0][a] = (*m_3)[0][a];
                    }
                }
                delete m_1;
                delete m_3;
            }

            return result;
            
        }

        Matrix feed_forward(std::vector<double> input_values){
            this->init(input_values);

            auto last_layer_size = this->layers[this->layers.size()-1].get_size();
            Matrix result(1, last_layer_size);
            
            for(uint i = 0; i < this->layers.size() - 1; ++i){
                auto m_1 = this->layers[i].get_value_matrix();
                auto m_2 = this->weights[i];
                auto m_3 = new Matrix(m_1->rows, m_2.cols);

                // std::cout << "Feed forward iteration: " << i << std::endl;
                // std::cout << " --- M_1 --- " << std::endl;
                // m_1->print();
                // std::cout << " --- M_2 --- " << std::endl;
                // // m_2.print();

                Matrix::multiply(m_1, &m_2, m_3);

                // std::cout << " --- M_3 --- " << std::endl;
                // m_3->print();

                for(uint j = 0; j < m_3->cols; ++j){
                    // auto layer = this->layers[i+1];
                    // auto val = (*m_3)[0][j];
                    // std::cout<<"net[" << i+1 << "][" << j << "]" << val << std::endl;
                    // layer.get_neuron(j).set_value(val);
                    auto vals = std::vector<double>(m_3->cols);
                    for(uint i = 0; i < m_3->cols; ++i){
                        vals[i] = (*m_3)[0][i];
                    }
                    this->layers[i+1].set_values(vals);
                }

                // If it's the last layer copy the calculated values to the matrix result
                if(i == this->layers.size()-2){
                    if(m_3->cols != result.cols){
                        throw "Implementation error!";
                    }

                    for(uint a = 0; a < last_layer_size; ++a){
                        result[0][a] = (*m_3)[0][a];
                    }
                }
                
                delete m_1;
                delete m_3;
            }
            std::cout << "Feed forward calculation done" << std::endl;
            return result;

        }

        void print(){
                std::cout << " ===== Layers ===== " << std::endl;
                for(uint i = 0; i < layers.size(); ++i){
                    std::cout << " --- Layer " << i << " --- " << std::endl;
                    (this->layers)[i].get_value_matrix()->print();
                }

                std::cout << " ===== Weights ===== " << std::endl;
                for(uint i = 0; i < layers.size()-1; ++i){
                    std::cout << " --- Weights " << i << " --- " << std::endl;
                    weights[i].print();
                }


        }

        ~NeuralNetwork(){
            // for(uint i = 0; i < layers.size(); ++i){
            //     delete layers[i];
            // }

            // for(uint i = 0; i < weights.size(); ++i){
            //     delete weights[i];
            // }

            // delete layers;

            // delete weights;
        }

    private:

        void init(std::vector<double> input_values){
             if(layers[0].get_size() != input_values.size()){
                throw std::invalid_argument("Size of input vector doesn't match topology of neural network!");
            }
            for(uint i = 0; i < input_values.size(); ++i){
                layers[0].get_neuron(i).init(input_values[i]);
            }
        }

        std::vector<Layer> layers;

        std::vector<Matrix> weights;
};