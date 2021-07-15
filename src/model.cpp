#include "model.h"
#include <iostream>
using namespace std;
nn::Model::Model(){


    
}


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

tensor nn::Model::predic(const tensor &x){
    tensor now = x;
    for(int i = 0; i < this->layers.size(); ++i){
        now = layers[i]->forward(now);
    }
    return now;
}