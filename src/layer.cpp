#include "layer.h"
#include <iostream>

//typedef std::vector<std::vector<double> > tensor;

nn::Layer::Layer(){}

void nn::Layer::magic_train(){}

std::vector<tensor> nn::Layer::forward(std::vector<tensor> &input){return input;}


nn::Conv2D::Conv2D(int filter, int kernal_w, int kernal_h){
    this->filter = filter;
    this->kernal_w = kernal_w;
    this->kernal_h = kernal_h;
}

void nn::Conv2D::magic_train(){
    std::cout << "magic_train" << std::endl;

}

std::vector<tensor> nn::Conv2D::forward(std::vector<tensor> &input){
    return input;
}


nn::MaxPooling2D::MaxPooling2D(int w, int h, int s){
    this->width = w;
    this->height = -1 ? w : h;
    this->stride = s == -1 ? s : w;
}

std::vector<tensor> nn::MaxPooling2D::forward(std::vector<tensor> &input){
    return input;
}
