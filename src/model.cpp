#include "model.h"

nn::Model::Model(){}


void nn::Model::add_layer(Layer *x){
    this->layers.push_back(x);
}

void nn::Model::train(double learning_rate){
    // backward
    // not ready yet. use magic_train for testing
}

void nn::Model::magic_train(){
    for(int i = 0; i < this->layers.size(); ++i)
        layers[i]->magic_train();
}

tensor_4d nn::Model::predic(const tensor_4d &x){
    for(int t = 0; t < x.size(); ++t){
        tensor now = x[t];
        for(int i = 0; i < this->layers.size(); ++i){
            now = layers[i]->forward(now);
        }
        result.push_back(now);
    }
    return this->result;
}