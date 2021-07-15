#include <vector>
#include <random>
// #include <time>

typedef  std::vector<std::vector<std::vector<std::vector<double> > > > tensor_4d;
typedef  std::vector<std::vector<std::vector<double> > > tensor;
//typedef  std::vector<std::vector<double> > tensor;

namespace nn{
    tensor new_tensor(int channel, int height, int width);
    tensor_4d new_tensor_4d(int filter, int channel, int height, int width);
    void random_tensor(tensor &x);
}