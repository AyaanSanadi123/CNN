#include <stdlib.h>
#include <stdio.h>
#include "core/tensor.h"

Tensor* create_tensor(int N, int H, int W, int C){
    if (N <= 0 || H <= 0 || W <= 0 || C <= 0) {
        fprintf(stderr, "Error: Tensor dimensions must be strictly positive.\n");
        return NULL;
    }

    Tensor* t = (Tensor*) malloc(sizeof(Tensor));
    if(!t){
        fprintf(stderr, "Error: Failed to allocate Tensor struct.\n");
        return NULL;
    }

    t->N = N;
    t->H = H;
    t->W = W;
    t->C = C;

    size_t total_elements = (size_t) N * H * W * C;

    // Allocate the flat array and initialize to exactly 0.0 (calloc is safer than malloc here)
    t->data = (float*)calloc(total_elements, sizeof(float));
    if (!t->data) {
        fprintf(stderr, "Error: Failed to allocate %zu floats for Tensor data.\n", total_elements);
        free(t);
        return NULL;
    }

    return t;
}

void tensor_free(Tensor* t){
    if(t){
        if(t -> data){
            free(t->data);
            t->data = NULL;
        }
        free(t);
    }
}

void tensor_fill(Tensor *t, float value) {
    if (!t || !t->data) return;
    
    size_t total_elements = (size_t)t->N * t->H * t->W * t->C;
    for (size_t i = 0; i < total_elements; i++) {
        t->data[i] = value;
    }
}