#include <iostream>
//#include <omp.h>
#include "model.h"
#include "tensor.h"

void print_tensor(tensor& x);

int main()
{
    nn::Model m = nn::Model();

    m.add_layer( new nn::Conv2D(4, 1, 3, 3) );
    m.add_layer( new nn::ReLU() );
    m.add_layer( new nn::Conv2D(1, 4, 3, 3, 1, 1, nn::SAME) );
    m.add_layer( new nn::ReLU() );
    //m.add_layer( new nn::MaxPooling2D(2) );

    // backward not ready yet. unable to train
    // using magic_train to assign random weights to each layer
    // m.train(0.0001);
    m.magic_train();

    // test input
    tensor test = nn::new_tensor(1, 8, 8);

    nn::random_tensor(test);

    print_tensor(test);

    tensor result = m.predic(test);

    print_tensor(result);

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