#include "layer.h"
#include <iostream>

nn::Layer::Layer(){}

void nn::Layer::magic_train(){}

tensor nn::Layer::forward(const tensor &input) {return input;}


nn::Conv2D::Conv2D(int filter, int channel, int kernal_w, int kernal_h, int stride_w, int stride_h, Padding padding){
    this->filter = filter;
    this->kernal_w = kernal_w;
    this->kernal_h = kernal_h;
    this->channel = channel;
    this->stride_w = stride_w;
    this->stride_h = stride_h;
    this->padding = padding;
    this->weight = new_tensor_4d(filter, channel, kernal_w, kernal_h);
    this->bias = std::vector<double>(filter);
}

void nn::Conv2D::magic_train(){
    //std::cout << "magic_train" << std::endl;
    random_tensor_4d(this->weight);
    random_tensor_1d(this->bias);
}

tensor nn::Conv2D::forward(const tensor &input) {
    int out_channel = input.size();
    int out_width = this->padding == SAME ? input[0].size() : input[0].size() - this->kernal_w + 1;
    int out_height =  this->padding == SAME ? input[0][0].size() : input[0][0].size() - this->kernal_h + 1;
    this->layer_out = new_tensor(filter, out_width, out_height);

    if (this->padding == VALID){ // no padding
        for(int i = 0; i < input[0].size() - kernal_w; i += this->stride_w)
            for(int j = 0; j < input[0][0].size() - kernal_h; j += this->stride_h)
                for(int f = 0; f < this->filter; ++f)
                    for(int c = 0; c < input.size(); ++c)
                        for(int m = 0; m < this->kernal_w; ++m)
                            for(int n = 0; n < this->kernal_h; ++n)
                                this->layer_out[f][i][j] += input[c][i+m][j+n] * weight[f][c][m][n];
    }
    else if (this->padding == SAME){  // pad to same size
        for(int i = 0; i < input[0].size(); i += this->stride_w)
            for(int j = 0; j < input[0][0].size(); j += this->stride_h)
                for(int f = 0; f < this->filter; ++f)
                    for(int c = 0; c < input.size(); ++c)
                        for(int m = 0; m < this->kernal_w; ++m)
                            for(int n = 0; n < this->kernal_h; ++n)
                                if (i + m >= input[0].size() || j + n > input[0][0].size())
                                    continue;
                                else
                                    this->layer_out[f][i][j] += input[c][i+m][j+n] * weight[f][c][m][n];
    }
    for(int f = 0; f < this->filter; ++f)
        for(int i = 0; i < this->layer_out[0].size(); i++)
            for(int j = 0; j < this->layer_out[0][0].size(); j++)
                this->layer_out[f][i][j] += this->bias[f];
    return this->layer_out;
}


nn::MaxPooling2D::MaxPooling2D(int w, int h, int s){
    this->width = w;
    this->height = h == -1 ? w : h;
    this->stride = s == -1 ? w : s;
}

tensor nn::MaxPooling2D::forward(const tensor &input) {
    int out_channel = input.size();
    int out_width = (input[0].size() - this->width) / this->stride + 1;
    int out_height = (input[0][0].size() - this->height) / this->stride + 1;
    this->layer_out = new_tensor(out_channel, out_width, out_height);

    for(int i = 0; i < out_channel; ++i){ //channel
         for(int j = 0; j < out_width; ++j){ // width
            for(int k = 0; k < out_height; ++k){ // height
                // maxpooling
                int x = j * this->stride, y = k * this->stride;
                for(int m = x; m < x + this->width; ++m)
                    for(int n = y; n < y + this->height; ++n)
                        this->layer_out[i][j][k] = std::max(this->layer_out[i][j][k], input[i][m][n]);
            } 
        }
    }
    return this->layer_out;
}

tensor nn::ReLU::forward(const tensor &input) {
    this->layer_out = new_tensor(input.size(), input[0].size(), input[0][0].size());
    for(int i = 0; i < input.size(); ++i)
        for(int j = 0; j < input[0].size(); ++j)
            for(int k = 0; k < input[0][0].size(); ++k)
                this->layer_out[i][j][k] = input[i][j][k] < 0 ? 0 : input[i][j][k];

    return this->layer_out;
}