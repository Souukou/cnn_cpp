#include <vector>
#include <omp.h>
#include "layer.h"
#include "tensor.h"


namespace nn{
    class Model{
    public:
        Model();
        void add_layer(Layer *x);
        void train(double learning_rate);
        void magic_train();
        tensor_4d predic(const tensor_4d &x, int threads = 0);
        tensor_4d result;
    private:
        std::vector<Layer*> layers;

    };
}
