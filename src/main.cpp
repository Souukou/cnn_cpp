#include <iostream>
//#include <omp.h>
#include "model.h"
#include "tensor.h"


int main()
{
    nn::Model m = nn::Model();

    m.add_layer( new nn::Conv2D(10, 1, 3, 3) );
    m.add_layer( new nn::ReLU() );    
    //m.add_layer( new nn::MaxPooling2D(2) );

    // backward not ready yet. unable to train
    // using magic_train to assign random weights to each layer
    // m.train(0.0001);
    m.magic_train();

    // test input
    tensor test = nn::new_tensor(1, 14, 14);

    nn::random_tensor(test);

    // for(int i = 0; i < test.size(); ++i){
    //     for(int j = 0; j < test[0].size(); ++j){
    //         for(int k = 0; k < test[0][0].size(); ++k)
    //             std::cout << test[i][j][k] << " ";
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }

    tensor result = m.predic(test);

    // for(int i = 0; i < result.size(); ++i){
    //     for(int j = 0; j < result[0].size(); ++j){
    //         for(int k = 0; k < result[0][0].size(); ++k)
    //             std::cout << result[i][j][k] << " ";
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }


    return 0;
}