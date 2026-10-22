#include "nn/layers.h"
#include <stdio.h>


void conv2d_forward(Tensor* input,Conv2DLayer *layer,Tensor *output){

    // output size of the activation maps, 
    // output size = (input_size - filter_size + (2*padding))/stride + 1
    int out_h = ((input->H - layer->filter_size + (2 * layer->padding)) / layer->stride) + 1;
    int out_w = ((input->W - layer->filter_size + (2 * layer->padding)) / layer->stride) + 1;

    if (output->N != input->N || output->H != out_h || output->W != out_w || output->C != layer->out_channels) {
        fprintf(stderr, "Error: Output tensor dimensions do not match Conv2D math.\n");
        return;
    }

    // the 7 loop sliding window 

    
}