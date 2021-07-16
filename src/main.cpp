#include <iostream>
#include "model.h"
#define INTEL
#ifdef INTEL
#include <mkl.h>
#endif
void print_tensor(tensor& x);
double get_time(){
#ifdef INTEL
    return dsecnd();
#endif
#ifndef INTEL
    return omp_get_wtime();
#endif
}
int main()
{
    nn::Model m = nn::Model();

    m.add_layer( new nn::Conv2D(4, 1, 3, 3) );
    m.add_layer( new nn::ReLU() );
    m.add_layer( new nn::MaxPooling2D(2) );
    m.add_layer( new nn::Conv2D(1, 4, 3, 3, 1, 1, nn::SAME) );
    m.add_layer( new nn::ReLU() );
    m.add_layer( new nn::MaxPooling2D(2) );

    // backward not ready yet. unable to train
    // using magic_train to assign random weights to each layer
    // m.train(0.0001);
    m.magic_train();

    // test input
    tensor_4d test = nn::new_tensor_4d(10000, 1, 16, 16);

    nn::random_tensor_4d(test);

    //print_tensor(test[0]);

    double start, end;
    int LOOP_COUNT = 5;
    tensor_4d result;
    // auto threads

    start = get_time();
    for (int i=0; i<LOOP_COUNT; ++i)
         result= m.predic(test);
    end = get_time();
    print_tensor(result[0]);

    std::cout << "Auto Threads Time Used: " << (end - start) / LOOP_COUNT << std::endl;

    // two threads
    start = get_time();
    for (int i=0; i<LOOP_COUNT; ++i)
        m.predic(test, 2);
    end = get_time();
    std::cout << "Two Threads Time Used: " << (end - start) / LOOP_COUNT << std::endl;

    // single threads (serialize)
    start = get_time();
    for (int i=0; i<LOOP_COUNT; ++i)
        m.predic(test, 1);
    end = get_time();
    std::cout << "Serialize Time Used: " << (end - start) / LOOP_COUNT << std::endl;


    return 0;
}

void print_tensor(tensor& x){
    for(int i = 0; i < x.size(); ++i){
        for(int j = 0; j < x[0].size(); ++j){
            for(int k = 0; k < x[0][0].size(); ++k)
                std::cout << x[i][j][k] << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}