#include "build/pinecone.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void printUsage() {
    printf("Usage: \n");
    printf("  To train: program_name train -i <input_csv> -o <output_model> -n <number_imgs> -l <hidden_layer_size> -r <learning_rate>\n");
    printf("Options:\n");
    printf("  -i <input_csv>        Path to the input CSV file with training data\n");
    printf("  -o <output_model>     Path to save the output model file\n");
    printf("  -n <number_imgs>      Number of images to use for training\n");
    printf("  -l <hidden_layer_size> Size of the hidden layer\n");
    printf("  -r <learning_rate>    Learning rate for training\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return 1;
    }

    // Default values
    char* input_csv = NULL;
    char* output_model = NULL;
    int number_imgs = 5000;             // Default value
    int hidden_layer_size = 1500;       // Default value
    double learning_rate = 0.0001;      // Default value

    if (strcmp(argv[1], "train") == 0) {
        // Parse flags
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                input_csv = argv[++i];
            } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
                output_model = argv[++i];
            } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                number_imgs = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-l") == 0 && i + 1 < argc) {
                hidden_layer_size = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
                learning_rate = atof(argv[++i]);
            } else {
                printUsage();
                return 1;
            }
        }

        // Ensure required arguments are provided
        if (!input_csv || !output_model) {
            fprintf(stderr, "Error: input file (-i) and output file (-o) are required.\n");
            printUsage();
            return 1;
        }

        // Load images and train the model
        Img** imgs = csvToImgs(input_csv, number_imgs);
        NeuralNetwork* net = networkCreate(784, hidden_layer_size, 10, learning_rate);
        printf("Training model with %d images, hidden layer size %d, learning rate %.4f\n",
               number_imgs, hidden_layer_size, learning_rate);
        
        networkTrainBatchImgs(net, imgs, number_imgs);

        // Save the trained model
        networkSave(net, output_model);
        printf("Model saved to %s\n", output_model);

        // Clean up
        imgsFree(imgs, number_imgs);
        networkFree(net);
    } else {
        printUsage();
    }

    return 0;
}
