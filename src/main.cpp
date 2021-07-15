#include <iostream>
#include "model.h"

void print_tensor(tensor& x);

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

    // auto threads
    double start = omp_get_wtime();
    tensor_4d result = m.predic(test);
    double end = omp_get_wtime();
    print_tensor(result[0]);
    std::cout << "Auto Threads Time Used: " << end - start << std::endl;

    // two threads
    start = omp_get_wtime();
    m.predic(test, 2);
    end = omp_get_wtime();
    std::cout << "Two Threads Time Used: " << end - start << std::endl;

    // single threads (serialize)
    start = omp_get_wtime();
    m.predic(test, 1);
    end = omp_get_wtime();
    std::cout << "Serialize Time Used: " << end - start << std::endl;


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