#include "nn/layers.h"
#include <stdio.h>

void conv2d_forward(Tensor *input, Conv2DLayer *layer, Tensor *output)
{
    // 1. Guardrail: Verify Input and Weight Depth
    if (input->C != layer->in_channels || layer->weights->C != layer->in_channels)
    {
        fprintf(stderr, "Error: Input depth (%d) or Weight depth (%d) does not match layer rules (%d).\n",
                input->C, layer->weights->C, layer->in_channels);
        return;
    }
    // output size of the activation maps,
    // output size = (input_size - filter_size + (2*padding))/stride + 1
    int out_h = ((input->H - layer->filter_size + (2 * layer->padding)) / layer->stride) + 1;
    int out_w = ((input->W - layer->filter_size + (2 * layer->padding)) / layer->stride) + 1;

    if (output->N != input->N || output->H != out_h || output->W != out_w || output->C != layer->out_channels)
    {
        fprintf(stderr, "Error: Output tensor dimensions do not match Conv2D math.\n");
        return;
    }

    // the 7 loop sliding window
    for (int b = 0; b < input->N; b++)
    { // this is the batch size, for now its handled sequentially
        for (int y = 0; y < out_h; y++)
        { // this is the height of the output
            for (int x = 0; x < out_w; x++)
            { // this is the width of the output
                for (int f = 0; f < layer->out_channels; f++)
                { // one filter from how many assigned
                    float dot_product = 0.0f;
                    // at this point, we are looking at a specific coordinate in the output activation map
                    // say the coordinate (0,0)
                    // now say out filter is 3 X 3, thats 9 pixels, those 9 values get crushed into 1 dot_product value in the activation map

                    // inside the filter,
                    for (int ky = 0; ky < layer->filter_size; ky++)
                    {
                        for (int kx = 0; kx < layer->filter_size; kx++)
                        {
                            // now we are inside the filter,
                            // we process each input pixel of the input and save it in the output
                            // for that we need to find the input pixel, becuase we are looking at this from the output's pov
                            int in_y = (y * layer->stride) + ky - layer->padding;
                            int in_x = (x * layer->stride) + kx - layer->padding;

                            if (in_y >= 0 && in_y < input->H && in_x >= 0 && in_x < input->W)
                            {
                                // make sure we are in bounds of the input image
                                for (int c = 0; c < input->C; c++)
                                {
                                    // Calculate the NHWC flat index for the INPUT image
                                    int in_idx = (b * input->H * input->W * input->C) +
                                                 (in_y * input->W * input->C) +
                                                 (in_x * input->C) +
                                                 c;

                                    // Calculate the NHWC flat index for the FILTER WEIGHTS
                                    int weight_idx = (f * layer->filter_size * layer->filter_size * input->C) +
                                                     (ky * layer->filter_size * input->C) +
                                                     (kx * input->C) +
                                                     c;

                                    // Multiply and accumulate
                                    dot_product += input->data[in_idx] * layer->weights->data[weight_idx];
                                }
                            }
                        }
                    }
                    // add bias for that specific filter
                    dot_product += layer->biases->data[f];
                    // Calculate the NHWC flat index for the OUTPUT image
                    int out_idx = (b * output->H * output->W * output->C) +
                                  (y * output->W * output->C) +
                                  (x * output->C) +
                                  f;

                    // Write the final feature value to memory
                    output->data[out_idx] = dot_product;

                    // so the filterSize * filterSize grid of pixels become just output data
                }
            }
        }
    }
}