#include "layer.h"
#include <iostream>

//typedef std::vector<std::vector<double> > tensor;

nn::Layer::Layer(){}

void nn::Layer::magic_train(){}

tensor nn::Layer::forward(tensor &input){return input;}


nn::Conv2D::Conv2D(int filter, int kernal_w, int kernal_h){
    this->filter = filter;
    this->kernal_w = kernal_w;
    this->kernal_h = kernal_h;
}

void nn::Conv2D::magic_train(){
    std::cout << "magic_train" << std::endl;

}

tensor nn::Conv2D::forward(tensor &input){
    return input;
}


nn::MaxPooling2D::MaxPooling2D(int w, int h, int s){
    this->width = w;
    this->height = h == -1 ? w : h;
    this->stride = s == -1 ? w : s;
}

tensor nn::MaxPooling2D::forward(tensor &input){
    int out_channel = input.size();
    int out_height = (input[0].size() - this->height) / this->stride + 1;
    int out_width = (input[0][0].size() - this->width) / this->stride + 1;
    tensor out_tensor = new_tensor(out_channel, out_height, out_width);

    for(int i = 0; i < out_channel; ++i){ //channel
         for(int j = 0; j < out_height; ++j){ // height
            for(int k = 0; k < out_width; ++k){ // width
                // maxpooling
                int x = j * this->stride, y = k * this->stride;
                for(int m = x; m < x + this->height; ++m)
                    for(int n = y; n < y + this->height; ++n)
                        out_tensor[i][j][k] = std::max(out_tensor[i][j][k], input[i][m][n]);
            } 
        }
    }
    return out_tensor;
}
