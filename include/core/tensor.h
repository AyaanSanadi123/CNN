#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

typedef struct {
    int N;       // Batch Size
    int H;       // Height
    int W;       // Width
    int C;       // Channels
    float *data; // Flat 1D contiguous block of heap memory
} Tensor;

Tensor* tensor_create(int N, int H, int W, int C);
void tensor_free(Tensor* t);
void tensor_fill(Tensor *t,float value); // this is used to initialize the bisas 

#endif