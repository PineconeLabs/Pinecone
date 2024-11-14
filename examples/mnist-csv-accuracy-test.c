#include "../build/pinecone.h"
#include <stdio.h>

int main() {
    int number_imgs = 3000;
	Img** imgs = csvToImg("data/mnist_test.csv", number_imgs);
	NeuralNetwork* net = nnLoad("testing_net");
	double score = nnPredictImgs(net, imgs, 1000);
	printf("Score: %1.5f\n", score);
}