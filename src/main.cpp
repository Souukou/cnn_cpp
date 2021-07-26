#include <iostream>
#include <iomanip>
#include "model.h"

//#define INTEL
#ifdef INTEL
#include <mkl.h>
#endif

void print_tensor(tensor& x);
double get_time();
void usage(char* prog_name);
void get_args(int argc, char* argv[], int &n, int &thread_count, bool &verbose);

void T1(){
    std::cout << "Test Case 1: Conv2D Only, 4 filter, 1 channel, 3x3 kernal, 1 stride, no padding" << std::endl;
    nn::Model m = nn::Model();
    m.add_layer( new nn::Conv2D(4, 1, 3, 3) );  // 4 filter, 1 channel, 3x3 kernal, 1 stride, no padding
    m.magic_train(1);
    tensor_4d test_tensor = nn::new_tensor_4d(1, 1, 5, 5);
    for(int i = 0; i < 5; ++i)
		for(int j = 0; j < 5; ++j)
            test_tensor[0][0][i][j] = i * 5 + j;
    //test_tensor = {{{1,2,3,4,5}, {6,7,8,9,10}, {11,12,13,14,15}, {16,17,18,19,20}, {21,22,23,24,25}}};
    std::cout << "Input Tensor:" << std::endl;
    print_tensor(test_tensor[0]);
    tensor_4d  result = m.predic(test_tensor, 1);
    std::cout << "Result: " << std::endl;
    print_tensor(result[0]);
    std::cout << "###########################" << std::endl;
}

void T2(){
    std::cout << "Test Case 2: Conv2D Only, 4 filter, 1 channel, 3x3 kernal, 2x2 stride, no padding" << std::endl;
    nn::Model m = nn::Model();
    m.add_layer( new nn::Conv2D(4, 1, 3, 3, 2, 2) );  // 4 filter, 1 channel, 3x3 kernal, 1 stride, no padding
    m.magic_train(1);
    tensor_4d test_tensor = nn::new_tensor_4d(1, 1, 5, 5);
    for(int i = 0; i < 5; ++i)
		for(int j = 0; j < 5; ++j)
            test_tensor[0][0][i][j] = i * 5 + j;
    //test_tensor = {{{1,2,3,4,5}, {6,7,8,9,10}, {11,12,13,14,15}, {16,17,18,19,20}, {21,22,23,24,25}}};
    std::cout << "Input Tensor:" << std::endl;
    print_tensor(test_tensor[0]);
    tensor_4d  result = m.predic(test_tensor, 1);
    std::cout << "Result: " << std::endl;
    print_tensor(result[0]);
    std::cout << "###########################" << std::endl;
}

void T3(){
    std::cout << "Test Case 3: Conv2D Only, 4 filter, 1 channel, 3x3 kernal, 1 stride, padding to same size" << std::endl;
    nn::Model m = nn::Model();
    m.add_layer( new nn::Conv2D(4, 1, 3, 3, 1, 1, nn::SAME) );  // 4 filter, 1 channel, 3x3 kernal, 1 stride, padding to same size
    m.magic_train(1);
    tensor_4d test_tensor = nn::new_tensor_4d(1, 1, 5, 5);
    for(int i = 0; i < 5; ++i)
		for(int j = 0; j < 5; ++j)
            test_tensor[0][0][i][j] = i * 5 + j;
    //test_tensor = {{{1,2,3,4,5}, {6,7,8,9,10}, {11,12,13,14,15}, {16,17,18,19,20}, {21,22,23,24,25}}};
    std::cout << "Input Tensor:" << std::endl;
    print_tensor(test_tensor[0]);
    tensor_4d  result = m.predic(test_tensor, 1);
    std::cout << "Result: " << std::endl;
    print_tensor(result[0]);
    std::cout << "###########################" << std::endl;
}

void T4(){
    std::cout << "Test Case 4: Conv2D + ReLU" << std::endl;
    nn::Model m = nn::Model();
    m.add_layer( new nn::Conv2D(4, 1, 3, 3, 1, 1, nn::SAME) );  // 4 filter, 1 channel, 3x3 kernal, 1 stride, padding to same size
    m.add_layer( new nn::ReLU() );
    m.magic_train(1);
    tensor_4d test_tensor = nn::new_tensor_4d(1, 1, 5, 5);
    for(int i = 0; i < 5; ++i)
		for(int j = 0; j < 5; ++j)
            test_tensor[0][0][i][j] = i * 5 + j;
    //test_tensor = {{{1,2,3,4,5}, {6,7,8,9,10}, {11,12,13,14,15}, {16,17,18,19,20}, {21,22,23,24,25}}};
    std::cout << "Input Tensor:" << std::endl;
    print_tensor(test_tensor[0]);
    tensor_4d  result = m.predic(test_tensor, 1);
    std::cout << "Result: " << std::endl;
    print_tensor(result[0]);
    std::cout << "###########################" << std::endl;
}

void T5(){
    std::cout << "Test Case 5: Conv2D + ReLU + MaxPooling2D" << std::endl;
    nn::Model m = nn::Model();
    m.add_layer( new nn::Conv2D(4, 1, 3, 3, 1, 1, nn::SAME) );  // 4 filter, 1 channel, 3x3 kernal, 1 stride, padding to same size
    m.add_layer( new nn::ReLU() );
    m.add_layer( new nn::MaxPooling2D(2) );
    m.magic_train(1);
    tensor_4d test_tensor = nn::new_tensor_4d(1, 1, 5, 5);
    for(int i = 0; i < 5; ++i)
		for(int j = 0; j < 5; ++j)
            test_tensor[0][0][i][j] = i * 5 + j;
    //test_tensor = {{{1,2,3,4,5}, {6,7,8,9,10}, {11,12,13,14,15}, {16,17,18,19,20}, {21,22,23,24,25}}};
    std::cout << "Input Tensor:" << std::endl;
    print_tensor(test_tensor[0]);
    tensor_4d  result = m.predic(test_tensor, 1);
    std::cout << "Result: " << std::endl;
    print_tensor(result[0]);
    std::cout << "###########################" << std::endl;
}

void T6(){
    std::cout << "Test Case 6: Performance Test" << std::endl;
    nn::Model m = nn::Model();
    m.add_layer( new nn::Conv2D(1, 4, 3, 3, 1, 1, nn::SAME) );
    m.add_layer( new nn::ReLU() );
    m.add_layer( new nn::MaxPooling2D(2) );
    // backward not ready yet. unable to train
    // using magic_train to assign random weights to each layer
    // m.train(0.0001);
    m.magic_train();
    // test input
    tensor_4d test_tensor = nn::new_tensor_4d(10000, 1, 28, 28);
    nn::random_tensor_4d(test_tensor);

    double start, end;
    int LOOP_COUNT = 5;
    double gflop, time_avg;
    tensor_4d result;

    gflop = 10000*((2*9)*28*28*4 + 28*28*4 + 28*4 + 27*4) * 1E-9;

    start = get_time();
    for (int i=0; i<LOOP_COUNT; ++i)
        result= m.predic(test_tensor, 1);
    end = get_time();

    time_avg =  (end - start) / LOOP_COUNT;
    std::cout << "1 Thread:" << std::endl << 
        "Time: " << time_avg << "  GFlop: " << gflop << "   GFlop/s: " << gflop/ time_avg << std::endl;

    start = get_time();
    for (int i=0; i<LOOP_COUNT; ++i)
         result= m.predic(test_tensor, 2);
    end = get_time();
    time_avg =  (end - start) / LOOP_COUNT;
    std::cout << "2 Thread:" << std::endl << 
        "Time: " << time_avg << "  GFlop: " << gflop << "   GFlop/s: " << gflop/ time_avg << std::endl;


    start = get_time();
    for (int i=0; i<LOOP_COUNT; ++i)
         result= m.predic(test_tensor, 4);
    end = get_time();
    time_avg =  (end - start) / LOOP_COUNT;
    std::cout << "4 Thread:" << std::endl << 
        "Time: " << time_avg << "  GFlop: " << gflop << "   GFlop/s: " << gflop/ time_avg << std::endl;

    // Test with I3-8100, 4C4T, 17.55GFlop/sec in theory
    // https://setiathome.berkeley.edu/cpu_list.php#:~:text=Intel(R)%20Core(TM)%20i3-8100,17.55
}

int main(int argc, char* argv[])
{

    //int thread_count = 0, n; // auto thread by default
    //bool verbose = false;
    //get_args(argc, argv, n, thread_count, verbose);

    // T1 - T5 are 5 test case used to verify the correctness of the code.
    T1();
    T2();
    T3();
    T4();
    T5();
    // T6 is performance test
    T6();

    return 0;
}

void print_tensor(tensor& x){
    for(int i = 0; i < x.size(); ++i){
        for(int j = 0; j < x[0].size(); ++j){
            for(int k = 0; k < x[0][0].size(); ++k)
                std::cout << std::setw(4) << x[i][j][k] << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

double get_time(){
#ifdef INTEL
    return dsecnd();
#endif
#ifndef INTEL
    return omp_get_wtime();
#endif
}

void usage(char* prog_name) {
    fprintf(stderr, "usage:   %s <n> <t> <v|q>\n", prog_name);
    fprintf(stderr, "   n:   number of inputs\n");
    fprintf(stderr, "   t:   number of threads, default auto thread\n");
    fprintf(stderr, "  'v':  verbose, print first tensor\n");
    fprintf(stderr, "  'q':  quiet\n");
} 

void get_args(int argc, char* argv[], int &n, int &thread_count, bool &verbose) {
    if (argc < 2 ) {
        usage(argv[0]);
        exit(0);
    }
    n = atoi(argv[1]);
    if (argc >= 3)
        thread_count = atoi(argv[2]);
    if (argc >= 4)
        verbose = *argv[3] == 'v';
    if (n <= 0 || thread_count < 0) {
        usage(argv[0]);
        exit(0);
    }
} 

