#include "../build/pinecone.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("starting training");
	int number_imgs = 10000;
	Img** imgs = csvToImgs("data/mnist_test.csv", number_imgs);
    if (imgs == NULL) {
        printf("Failed to load images from CSV.\n");
        return 1;
    }

	NeuralNetwork* net = nnCreate(784, 300, 10, 0.1);
    nnTrainBatchImgs(net, imgs, number_imgs);
    printf("Training completed.\n");

	nnSave(net, "testing_net");
}
