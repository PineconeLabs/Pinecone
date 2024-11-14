#include "build/pinecone.h"
#include <stdio.h>


int main() {
    int number_imgs = 10000;
	Img** imgs = csv_to_imgs("data/mnist_test.csv", number_imgs);
	NeuralNetwork* net = network_load("mnist-30k-1000-0~001");
	double score = network_predict_imgs(net, imgs, 1000);
	printf("Score: %1.5f\n", score);
}