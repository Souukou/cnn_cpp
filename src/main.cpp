#include <iostream>
#include "model.h"

#define INTEL
#ifdef INTEL
#include <mkl.h>
#endif

void print_tensor(tensor& x);
double get_time();
void usage(char* prog_name);
void get_args(int argc, char* argv[], int &n, int &thread_count, bool &verbose);

int main(int argc, char* argv[])
{
    int thread_count = 0, n; // auto thread by default
    bool verbose = false;
    get_args(argc, argv, n, thread_count, verbose);

    nn::Model m = nn::Model();

    m.add_layer( new nn::Conv2D(4, 1, 3, 3) );
    m.add_layer( new nn::ReLU() );
    m.add_layer( new nn::MaxPooling2D(2) );
    m.add_layer( new nn::Conv2D(1, 4, 3, 3, 1, 1, nn::SAME) );
    m.add_layer( new nn::ReLU() );
    m.add_layer( new nn::MaxPooling2D(2) );

    // backward not ready yet. unable to train
    // using magic_train to assign random weights to each layer
    // m.train(0.0001);
    m.magic_train();

    // test input
    tensor_4d test = nn::new_tensor_4d(n, 1, 16, 16);

    nn::random_tensor_4d(test);
    
    if(verbose)
        print_tensor(test[0]);

    double start, end;
    int LOOP_COUNT = 5;
    tensor_4d result;


    start = get_time();
    for (int i=0; i<LOOP_COUNT; ++i)
         result= m.predic(test, thread_count);
    end = get_time();

    if(verbose)
        print_tensor(result[0]);

    std::cout << "Auto Threads Time Used: " << (end - start) / LOOP_COUNT << std::endl;


    return 0;
}

void print_tensor(tensor& x){
    for(int i = 0; i < x.size(); ++i){
        for(int j = 0; j < x[0].size(); ++j){
            for(int k = 0; k < x[0][0].size(); ++k)
                std::cout << x[i][j][k] << " ";
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

