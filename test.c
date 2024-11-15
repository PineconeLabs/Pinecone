#include "build/pinecone.h"
#include <stdio.h>


int main() {
    int number_imgs = 10000;
	Img** imgs = csvToImgs("data/mnist_test.csv", number_imgs);
	NeuralNetwork* net = networkLoad("mnist-60k");
	double score = networkPredictImgs(net, imgs, number_imgs);
	printf("Score: %1.5f\n", score);
}