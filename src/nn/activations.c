#include "nn/layers.h"

void leaky_relu_forward(Tensor *tensor, float alpha) {
    // 1. Calculate the absolute total number of floats in the memory block
    int total_elements = tensor->N * tensor->H * tensor->W * tensor->C;
    
    // 2. A single 1D loop marching straight across the RAM
    for (int i = 0; i < total_elements; i++) {
        
        float val = tensor->data[i];
        
        // 3. The Leaky ReLU formula
        if (val > 0.0f) {
            tensor->data[i] = val;         // Pass positive numbers through unchanged
        } else {
            tensor->data[i] = val * alpha; // Scale negative numbers down (e.g., alpha = 0.01)
        }
    }
}