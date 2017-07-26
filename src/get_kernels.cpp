#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace std;


/*
 * run:
 * ./classify PATH_TO_LENET_MODEL PATH_TO_MNIST_DATASET
 */

void usage() {
    std::cout << "Usage: ./classify PATH_TO_LENET_MODEL" << std::endl;
}

void get_kernels(const std::string &dictionary) {
    network<sequential> nn;
    nn.load(dictionary);

    //for (int i = 0; i < nn.depth(); i++) {
    for (int i = 0; i < 8; i++) {
        cout << "#layer:" << i << "\n";
        cout << "layer type:" << nn[i]->layer_type() << "\n";
        //cout << "input:" << nn[i]->in_size() << "(" << nn[i]->in_shape() << ")\n";
        //cout << "output:" << nn[i]->out_size() << "(" << nn[i]->out_shape() << ")\n";
        std::vector<shape3d> in_shape = nn[i]->in_shape();
        std::vector<shape3d> out_shape = nn[i]->out_shape();
       
        /*
        do not delete
        for(int j = 0; j < in_shape.size(); j++) {
            cout << in_shape[j].width_ << "x" << in_shape[j].height_ << "x" << in_shape[j].depth_ << "\n";
        }
        */

        if(nn[i]->layer_type().compare("conv") == 0) {
            cout << "number of input images:     " << in_shape[0].depth_ << "\n";
            cout << "input image height x width: " << in_shape[0].height_ << "x" << in_shape[0].width_ << "\n";      
            cout << "number of output images:    " << out_shape[0].depth_ << "\n";
            cout << "output image height x width:" << out_shape[0].height_ << "x" << out_shape[0].width_ << "\n";      
            cout << "kernel height x width:      " << in_shape[1].height_ << "x" << in_shape[1].width_ << "\n"; 
            cout << "number of kernels:          " << in_shape[1].depth_ << "\n";
            cout << "number of biases:           " << in_shape[2].depth_ << "\n";

            /*
            cout << "kernels:" << "\n";
            int kernel_height = in_shape[1].height_;
            int kernel_width = in_shape[1].width_;
            int num_kernels = in_shape[1].depth_;

            for(int j = 0; j < num_kernels; j++) {
                vec_t &weights = *nn[i]->weights()[0];
                
                for(int row = 0; row < kernel_height; row++) {
                    for(int column = 0; column < kernel_width; column++) {
                        cout << weights[j*kernel_height*kernel_width + row*kernel_width + column] << ", ";
                    }
                    cout << "\n";
                }
                cout << "\n";
            }

            cout << "bias:" << "\n";
            int num_biasses = in_shape[2].depth_;
            vec_t &bias = *nn[i]->weights()[1];

            for(int j = 0; j < num_biasses; j++) {
                cout << bias[j] << ", ";
            }
            */
        }

        if(nn[i]->layer_type().compare("max-pool") == 0) {
            cout << "number of input images:     " << in_shape[0].depth_ << "\n";
            cout << "input image height x width: " << in_shape[0].height_ << "x" << in_shape[0].width_ << "\n";      
            cout << "number of output images:    " << out_shape[0].depth_ << "\n";
            cout << "output image height x width:" << out_shape[0].height_ << "x" << out_shape[0].width_ << "\n";      
            cout << "kernel height x width:      " << in_shape[1].height_ << "x" << in_shape[1].width_ << "\n";
        }

        cout << "\n";
        
    }

    /*
    vec_t &W = *nn[0]->weights()[0];
    vec_t &B = *nn[0]->weights()[1];

    // 25*6 = 150
    for(int i = 0; i < W.size(); i++) {
        cout << W[i] << "\n";
    }

    cout << B.size() << "\n";
    for(int i = 0; i < B.size(); i++) {
        cout << B[i] << "\n";
    }
    */
}

int main(int argc, char **argv) {
    if(argc != 2) {
        usage();
        return 1;
    }

    get_kernels(argv[1]);

    return 0;
}
