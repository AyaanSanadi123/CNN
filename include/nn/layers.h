#ifndef LAYERS_H
#define LAYERS_H
#include "core/tensor.h"

typedef struct {
    int filter_size;
    int in_channels;
    int out_channels;
    int stride;
    int padding;
    Tensor* weights;
    Tensor* biases;
} Conv2DLayer;

typedef struct {
    int pool_size;
    int stride;
} MaxPool2DLayer;

typedef struct {
    int channels;          // Matches the C dimension of the input tensor
    float epsilon;         // Small number (e.g., 1e-5) to prevent divide-by-zero
    Tensor *gamma;         // Learnable scale factor [1, 1, 1, channels]
    Tensor *beta;          // Learnable shift factor [1, 1, 1, channels]
    Tensor *running_mean;  // Historical average [1, 1, 1, channels]
    Tensor *running_var;   // Historical variance [1, 1, 1, channels]
} BatchNorm2DLayer;

// write the forward pass functions 
void conv2d_forward(Tensor* input,Conv2DLayer *layer,Tensor* output);

void batchnorm2d_forward(Tensor* input,BatchNorm2DLayer* layer,Tensor* output);

void leaky_relu_forward(Tensor* tensor,float alpha);

void maxpool2d_forward(Tensor* input,MaxPool2DLayer*layer,Tensor* output);

#endif