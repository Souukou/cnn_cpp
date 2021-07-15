#include <vector>
#include <random>
// #include <time>

typedef  std::vector<std::vector<std::vector<double> > > tensor;
//typedef  std::vector<std::vector<double> > tensor;

namespace nn{
    tensor new_tensor(int channel, int height, int width);
    void random_tensor(tensor &x);
}