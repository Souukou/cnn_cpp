#include "model.h"

nn::Model::Model(){}


void nn::Model::add_layer(Layer *x){
    this->layers.push_back(x);
}

void nn::Model::train(double learning_rate){
    // backward
    // not ready yet. use magic_train for testing
}

void nn::Model::magic_train(int flag){
    for(int i = 0; i < this->layers.size(); ++i)
        layers[i]->magic_train(flag);
}

tensor_4d nn::Model::predic(const tensor_4d &x, int threads){
    this->result = std::vector<tensor>(x.size());
    int t;
    #pragma omp parallel for num_threads(threads)
    for(t = 0; t < x.size(); ++t){
        tensor now = x[t];
        for(int i = 0; i < this->layers.size(); ++i){
            now = layers[i]->forward(now);
        }
        result[t] = now;
    }
    return this->result;
}