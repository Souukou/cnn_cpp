# cnn_cpp

 卷积神经网络的C++实现，并使用OpenMP进行并行计算。

---

## 编译运行

### Intel芯片

1. 安装Intel oneAPI

   注意Intel oneAPI的安装程序**只能在图形界面下安装**，SSH连接必须要支持X11-forward，经测试无法在VSCode SSH终端中安装。(可用Terminus+VcXsrv)。也可以用APT安装，但是无法自定义安装的组件。

   ```bash
   cd ~
   wget https://registrationcenter-download.intel.com/akdlm/irc_nas/17977/l_BaseKit_p_2021.3.0.3219_offline.sh
   sudo bash l_BaseKit_p_2021.3.0.3219_offline.sh
   
   # 激活编译器环境
   . /opt/intel/oneapi/setvars.sh
   ```

2. 编译运行

   使用Intel编译器编译

   ```
   # icpx 相当于Intel版的g++, -lmkl_rt -osgemm_icc启用MKL库
   icpx -qopenmp -lmkl_rt -osgemm_icc src/*.cpp -o build/main
   ```

   使用Intel编译器编译并启用IPO优化

   ```
   icpx -qopenmp -lmkl_rt -osgemm_icc -ipo src/*.cpp -o build/main
   ```


### 其它芯片

需要去除#define INTEL，然后编译运行即可

```bash
g++ -fopenmp src/*.cpp -o build/main
```

### 运行

```
./build/main <n> <t> <v|q>
   n:   number of inputs
   t:   number of threads, default auto thread
  'v':  verbose, print first tensor
  'q':  quiet
```

## 主要实现的功能

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

## 测试结果

### 测试环境

Intel i3-8100 4C/4T

### 运行数据

1. 使用Intel编译器与gcc编译器的对比（单线程）

   | n      | GCC编译器 | Intel编译器 |
   | ------ | --------- | ----------- |
   | 1000   | 0.291444  | 0.0291      |
   | 10000  | 2.91841   | 0.2929      |
   | 100000 | 29.2933   | 3.03301     |

2. 使用Intel编译器，不开启IPO优化与开启IPO优化的对比

   | n      | 不开启IPO优化 | 开启IPO优化 |
   | ------ | ------------- | ----------- |
   | 1000   | 0.0291        | 0.0244      |
   | 10000  | 0.2929        | 0.2513      |
   | 100000 | 3.03301       | 2.5676      |

3. 使用Intel编译器，使用OpenMP多线程的对比
   | n      | 串行 | 2线程  | 2线程加速比 | 4线程  | 4线程加速比 |
   | ------ | ------------- | ----------- | ----------- | ----------- | ----------- |
   | 1000   | 0.0245 | 0.0137 | 1.78 | 0.0079 | 3.10 |
   | 10000  | 0.2486 | 0.1386 | 1.79 | 0.0779 | 3.19 |
   | 100000 | 2.5676 | 1.3932 | 1.84 | 0.7535 | 3.40 |

## TODO


3. gflops计算

4. 添加conv2d_fused

