**# cnn_cpp**

 卷积神经网络的C++实现，并使用OpenMP进行并行计算。

---

**# 编译运行**

**### Intel芯片，安装Intel oneAPI**

注意Intel oneAPI的安装程序**只能在图形界面下安装**，SSH连接必须要支持X11-forward，经测试无法在VSCode SSH终端中安装。(可用Terminus+VcXsrv)

```bash
cd ~
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/17977/l_BaseKit_p_2021.3.0.3219_offline.sh
sudo bash l_BaseKit_p_2021.3.0.3219_offline.sh

. /opt/intel/oneapi/setvars.sh

cd cnn_cpp
# icpx 相当于Intel版的g++, -lmkl_rt -osgemm_icc启用MKL库
icpx -qopenmp -lmkl_rt -osgemm_icc src/*.cpp -o build/main
```

**###其它芯片**

需要去除#define INTEL，然后编译运行即可

```bash
g++ -fopenmp src/*.cpp -o build/main
./build/main
```

**#主要实现的功能**

```c++
Conv2D(
    int filter,               // 卷积核数
    int channel,              // 输入向量通道数
    int kernal_w,             // 卷积核宽度
    int kernal_h,             // 卷积核高度
    int stride_w = 1,         // 宽方向的步长，默认为1
    int stride_h = 1,         // 高方向的步长，默认为1
    Padding padding=VALID     // 填充 nn::VALID不填充 nn::SAME填充使得输出与输入大小相同 默认不填充
);
```

```c++
MaxPooling2D(
    int w,                    // 池化宽度
    int h = -1,               // 池化高度，默认等于宽度w
    int s = -1                // 池化步长，默认等于宽度w
);
```

```c++
ReLU()                        // ReLU激活函数层
```

```c++
Model.add_layer(Layer *x)  // 向模型中添加层
Model.magic_train()        // 给每个层的权重随机初始化，用于测试前向传播
Model.predic(              // 前向传播
    const tensor_4d &x,    // 输入格式（图像, 通道, x, y）
    int threads = 0        // 默认自动线程数
)
```

**#测试结果**



  