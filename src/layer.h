#include <vector>
#include "tensor.h"

namespace nn{
    typedef enum{
        VALID, SAME
    } Padding;


    class Layer{
    public:
        Layer();
        virtual void magic_train();
        virtual tensor forward(const tensor &input);
        tensor layer_out;
    };


    class Conv2D : public Layer{
    public:
        Conv2D(int filter, int channel, int kernal_w, int kernal_h, int stride_w = 1, int stride_h = 1, Padding padding=VALID);
        ~Conv2D();
        virtual void magic_train();
        virtual tensor forward(const tensor &input);
    private:
        int filter, channel, kernal_w, kernal_h, stride_w, stride_h;
        Padding padding;
        std::vector<tensor> weight;
        std::vector<double> bias;
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

