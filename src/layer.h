#include <vector>
#include "tensor.h"

namespace nn{

    class Layer{
    public:
        Layer();
        virtual void magic_train();
        virtual tensor forward(const tensor &input);
        tensor layer_out;
    };


    class Conv2D : public Layer{
    public:
        Conv2D(int filter, int kernal_w, int kernal_h);
        ~Conv2D();
        virtual void magic_train();
        virtual tensor forward(const tensor &input);
    private:
        int filter, kernal_w, kernal_h;
        std::vector <tensor> array;
    };


    class MaxPooling2D : public Layer{
    public:
        MaxPooling2D(int w, int h = -1, int s = -1);
        virtual tensor forward(const tensor &input);
    private:
        int width, height, stride;  // stride = height = width by default
        
    };


    class ReLU : public Layer{
    public:
        virtual tensor forward(const tensor &input);
    };
}

