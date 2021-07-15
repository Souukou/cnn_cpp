#include "tensor.h"
#include <iostream>
tensor nn::new_tensor(int height, int width){
    //tensor x(channel, std::vector<std::vector<int> >(height, std::vector<int>(width)));
    tensor x(height, std::vector<double>(width));
    return x;
}

void nn::random_tensor(tensor &x){
    int col = x.size();
    int row = x[0].size();
    //std::cout << col << "  " << row << std::endl;
    std::default_random_engine e;
	std::uniform_real_distribution<double> u(0, 1);
    e.seed(time(0));
    for(int i = 0; i < col; ++i)
        for(int j = 0; j < row; ++j)
            x[i][j] = u(e);
        
}
