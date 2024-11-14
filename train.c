#include "build/pinecone.h"
#include <stdio.h>

void train_and_save_model(int number_imgs, int hiddenLayerSize, double learningRate) {
    // Load training data
    Img** imgs = csv_to_imgs("./data/mnist_train.csv", number_imgs);

    // Create and train the network
    NeuralNetwork* net = network_create(784, hiddenLayerSize, 10, learningRate);
    printf("Training mnist-%dk-%d-%.4f\n", number_imgs / 1000, hiddenLayerSize, learningRate);
    network_train_batch_imgs(net, imgs, number_imgs);

    // Save the trained model
    char filename[50];
    snprintf(filename, sizeof(filename), "mnist-models/mnist-%dk-%d-%.4f", number_imgs / 1000, hiddenLayerSize, learningRate);
    network_save(net, filename);

    // Clean up
    imgs_free(imgs, number_imgs);
    network_free(net);
}

int main() {
    int hiddenLayerSize = 1500;
    double learningRate = 0.0001;

    // Train models on progressively larger subsets of the MNIST dataset
    for (int number_imgs = 5000; number_imgs <= 60000; number_imgs += 5000) {
        train_and_save_model(number_imgs, hiddenLayerSize, learningRate);
    }

    return 0;
}
