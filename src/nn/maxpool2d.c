#include "nn/layers.h"
#include <stdio.h>
#include <float.h>

void maxpool2d_forward(Tensor *input, MaxPool2DLayer *layer, Tensor *output){
    // 1. Calculate expected output dimensions (No padding in standard MaxPool)
    int out_h = ((input->H - layer->pool_size) / layer->stride) + 1;
    int out_w = ((input->W - layer->pool_size) / layer->stride) + 1;

    // 2. Guardrail: Verify output bucket geometry and depth
    if (output->N != input->N || output->H != out_h || output->W != out_w || output->C != input->C) {
        fprintf(stderr, "Error: Output tensor dimensions do not match MaxPool2D math.\n");
        return;
    }

    for (int b = 0; b < input->N; b++) {                             // 1. Image in batch
        for (int y = 0; y < out_h; y++) {                            // 2. Output Y coordinate
            for (int x = 0; x < out_w; x++) {                        // 3. Output X coordinate
                for (int c = 0; c < input->C; c++) {                 // 4. Color / Feature Channel
                    
                    // Start with the lowest possible float value
                    float max_val = -FLT_MAX; 
                    
                    // 5 & 6. The 2x2 Spatial Window
                    for (int ky = 0; ky < layer->pool_size; ky++) {
                        for (int kx = 0; kx < layer->pool_size; kx++) {
                            
                            // Map back to the input image coordinates
                            int in_y = (y * layer->stride) + ky;
                            int in_x = (x * layer->stride) + kx;
                            
                            // Bounds checking (in case the window hangs off the edge)
                            if (in_y < input->H && in_x < input->W) {
                                
                                // Calculate the NHWC flat index for the INPUT image
                                int in_idx = (b * input->H * input->W * input->C) + 
                                             (in_y * input->W * input->C) + 
                                             (in_x * input->C) + 
                                             c;
                                             
                                float current_val = input->data[in_idx];
                                
                                // Keep only the largest value found in the window
                                if (current_val > max_val) {
                                    max_val = current_val;
                                }
                            }
                        }
                    }
                    
                    // Calculate the NHWC flat index for the OUTPUT image
                    int out_idx = (b * output->H * output->W * output->C) + 
                                  (y * output->W * output->C) + 
                                  (x * output->C) + 
                                  c;
                                  
                    // Write the maximum value found to the compressed memory block
                    output->data[out_idx] = max_val;
                }
            }
        }
    }
}