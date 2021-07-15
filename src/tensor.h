#include <vector>
#include <random>
// #include <time>

//typedef  std::vector<std::vector<std::vector<int> > > tensor;
typedef  std::vector<std::vector<double> > tensor;

namespace nn{
    tensor new_tensor(int height, int width);
    void random_tensor(tensor &x);
}