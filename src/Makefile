# TODO use nnpack for multi-core CPU
# TODO use libdnn for OpenCL GPU

CC=gcc-4.8
TINY_DNN=/Users/konze/Programming/tiny-dnn
CFLAGS=-DCNN_USE_AVX -DCNN_USE_SSE -DUSE_OPENCL -DDNN_USE_IMAGE_API -I$(TINY_DNN) -pthread -msse3 -mavx -lstdc++ -std=gnu++11 -O2 

all: train test

train: train.cpp
	$(CC) $(CFLAGS) train.cpp -o train -framework OpenCL

test: test.cpp
	$(CC) $(CFLAGS) test.cpp -o test -framework OpenCL

.PHONY clean: 
	-rm train
	-rm test
