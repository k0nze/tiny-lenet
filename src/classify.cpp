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
    std::cout << "Usage: ./classify PATH_TO_LENET_MODEL PATH_TO_MNIST_DATASET"  << std::endl;
}

void recognize(const std::string &dictionary, const std::string &mnist_dataset_path) {
    network<sequential> nn;
    nn.load(dictionary);

    // load MNIST dataset
    std::vector<label_t> test_labels;
    std::vector<vec_t> test_images;

    parse_mnist_labels(mnist_dataset_path + "/t10k-labels.idx1-ubyte", &test_labels);
    parse_mnist_images(mnist_dataset_path + "/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 2, 2);

    nn.test(test_images, test_labels).print_detail(std::cout);
}

int main(int argc, char **argv) {
    if(argc != 3) {
        usage();
        return 1;
    }

    recognize(argv[1], argv[2]);

    return 0;
}
