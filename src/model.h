//#include <vector>
#include "layer.h"
#include "tensor.h"

namespace nn{
    class Model{
    public:
        Model();
        void add_layer(Layer *x);
        void train(double learning_rate);
        void magic_train();
        tensor predic(const tensor &x);
    private:
        std::vector<Layer*> layers;

    };
}
