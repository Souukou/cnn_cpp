#include <iostream>
//#include <omp.h>
#include "model.h"
#include "tensor.h"


int main()
{
    nn::Model m = nn::Model();
    nn::Layer *layer1 = new nn::Conv2D(10, 3, 3);
    nn::Layer *layer2 = new nn::MaxPooling2D(2);
    m.add_layer(layer1);    
    m.add_layer(layer2);

    // no backward, using magic_train to assign random weights to each layer
    // m.train(0.0001);
    m.magic_train();

    // test input
    tensor test = nn::new_tensor(128, 28);
    nn::random_tensor(test);

    // for(int i = 0; i < test.size(); ++i){
    //     for(int j = 0; j < test[0].size(); ++j)
    //         std::cout << test[i][j] << " ";
    //     std::cout<<std::endl;
    // }
    

    tensor result = m.predic(test);


    return 0;
}