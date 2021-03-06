#include "tensor.h"

tensor nn::new_tensor(int channel, int height, int width){
    tensor x(channel, std::vector<std::vector<double> >(height, std::vector<double>(width)));
    //tensor x(height, std::vector<double>(width));
    return x;
}

tensor_4d nn::new_tensor_4d(int filter, int channel, int height, int width){
    tensor_4d x( filter, std::vector<std::vector<std::vector<double> > >(channel, std::vector<std::vector<double> >(height, std::vector<double>(width) ) ) );
    return x;
}

void nn::random_tensor(tensor &x){
    int channel = x.size();
    int height = x[0].size();
    int width = x[0][0].size();
    std::default_random_engine e;
	std::uniform_real_distribution<double> u(-1, 1);
    e.seed(time(0));
    for(int i = 0; i < channel; ++i)
        for(int j = 0; j < height; ++j)
            for(int k = 0; k < width; ++k)
                x[i][j][k] = u(e);
        
}

void nn::random_tensor_4d(tensor_4d &x){
    int filter = x.size();
    int channel = x[0].size();
    int height = x[0][0].size();
    int width = x[0][0][0].size();
    std::default_random_engine e;
	std::uniform_real_distribution<double> u(0, 1);
    e.seed(time(0));
    for(int f = 0; f < filter; ++f)
        for(int i = 0; i < channel; ++i)
            for(int j = 0; j < height; ++j)
                for(int k = 0; k < width; ++k)
                    x[f][i][j][k] = u(e);
        
}

void nn::random_tensor_1d(std::vector<double> &x){
    int channel = x.size();
    std::default_random_engine e;
	std::uniform_real_distribution<double> u(0, 1);
    e.seed(time(0));
    for(int i = 0; i < channel; ++i)
        x[i] = u(e);
        
}