#include <vector>
#include "tensor.h"

namespace nn{

    class Layer{
    public:
        Layer();
        virtual void magic_train();
        virtual std::vector<tensor> forward(std::vector<tensor> &input);
    };


    class Conv2D : public Layer{
    public:
        Conv2D(int filter, int kernal_w, int kernal_h);
        ~Conv2D();
        virtual void magic_train();
        virtual std::vector<tensor> forward(std::vector<tensor> &input);
    private:
        int filter, kernal_w, kernal_h;
        std::vector <tensor> array;
    };


    class MaxPooling2D : public Layer{
    public:
        MaxPooling2D(int w, int h = -1, int s = -1);
        virtual std::vector<tensor> forward(std::vector<tensor> &input);
    private:
        int width, height, stride;  // stride = height = width by default
        
    };
}
