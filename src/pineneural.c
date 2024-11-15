#include "pinecone.h"
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define MAXCHAR 1000

// 784, 300, 10
NeuralNetwork* networkCreate(int input, int hidden, int output, double lr) {
    NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
    net->input = input;
    net->hidden = hidden;
    net->output = output;
    net->learningRate = lr;

    Matrix* hiddenLayer = matrixCreate(hidden, input);
    Matrix* outputLayer = matrixCreate(output, hidden);
    matrixRandomize(hiddenLayer, hidden);
    matrixRandomize(outputLayer, output);

    net->hiddenWeights = hiddenLayer;
    net->outputWeights = outputLayer;

    return net;
}

void networkTrain(NeuralNetwork* net, Matrix* input, Matrix* output) {
    // Feed forward
    Matrix* hiddenInputs = matrixDot(net->hiddenWeights, input);
    Matrix* hiddenOutputs = matrixApply(sigmoid, hiddenInputs);
    Matrix* finalInputs = matrixDot(net->outputWeights, hiddenOutputs);
    Matrix* finalOutputs = matrixApply(sigmoid, finalInputs);

    // Find errors
    Matrix* outputErrors = matrixSubtract(output, finalOutputs);
    Matrix* transposedMat = matrixTranspose(net->outputWeights);
    Matrix* hiddenErrors = matrixDot(transposedMat, outputErrors);
    matrixFree(transposedMat);

    // Backpropogate
    Matrix* sigmoidPrimedMat = matrixSigmoidPrime(finalOutputs);
    Matrix* multipliedMat = matrixMultiply(outputErrors, sigmoidPrimedMat);
    transposedMat = matrixTranspose(hiddenOutputs);
    Matrix* dotMat = matrixDot(multipliedMat, transposedMat);
    Matrix* scaledMat = matrixScale(net->learningRate, dotMat);
    Matrix* addedMat = matrixAdd(net->outputWeights, scaledMat);

    matrixFree(net->outputWeights); // Free the old weights before replacing
    net->outputWeights = addedMat;

    matrixFree(sigmoidPrimedMat);
    matrixFree(multipliedMat);
    matrixFree(transposedMat);
    matrixFree(dotMat);
    matrixFree(scaledMat);

    // hiddenWeights update
    sigmoidPrimedMat = matrixSigmoidPrime(hiddenOutputs);
    multipliedMat = matrixMultiply(hiddenErrors, sigmoidPrimedMat);
    transposedMat = matrixTranspose(input);
    dotMat = matrixDot(multipliedMat, transposedMat);
    scaledMat = matrixScale(net->learningRate, dotMat);
    addedMat = matrixAdd(net->hiddenWeights, scaledMat);
    matrixFree(net->hiddenWeights); // Free the old hiddenWeights before replacement
    net->hiddenWeights = addedMat;

    matrixFree(sigmoidPrimedMat);
    matrixFree(multipliedMat);
    matrixFree(transposedMat);
    matrixFree(dotMat);
    matrixFree(scaledMat);

    // Free matrices
    matrixFree(hiddenInputs);
    matrixFree(hiddenOutputs);
    matrixFree(finalInputs);
    matrixFree(finalOutputs);
    matrixFree(outputErrors);
    matrixFree(hiddenErrors);
}

#include <stdio.h>

// Your original function with progress bar addition
#include <stdio.h>
#include <stdio.h>
#include <stdio.h>

#include <stdio.h>

void networkTrainBatchImgs(NeuralNetwork* net, Img** imgs, int batchSize) {
    int lastPercentage = -1;

    for (int i = 0; i < batchSize; i++) {
        int percentage = (i * 100) / batchSize;

        if (percentage != lastPercentage || i == batchSize - 1) {
            printf("\r%-50s\rTraining progress: %d%% - Processing image %d of %d",
                   "", percentage, i + 1, batchSize);
            fflush(stdout);
            lastPercentage = percentage;
        } else {
            printf("\r%-50s\rTraining progress: %d%% - Processing image %d of %d",
                   "", percentage, i + 1, batchSize);
            fflush(stdout);
        }

        Img* curImg = imgs[i];
        Matrix* imgData = matrixFlatten(curImg->imgData, 0);
        Matrix* output = matrixCreate(10, 1);
        output->entries[curImg->label][0] = 1;  

        networkTrain(net, imgData, output);

        // Free memory
        matrixFree(output);
        matrixFree(imgData);
    }

    printf("\r%-50s\rTraining progress: 100%% - Processed all %d images\n", "", batchSize);
    fflush(stdout);
}




Matrix* networkPredictImg(NeuralNetwork* net, Img* img) {
    Matrix* imgData = matrixFlatten(img->imgData, 0);
    Matrix* res = networkPredict(net, imgData);
    matrixFree(imgData);
    return res;
}

double networkPredictImgs(NeuralNetwork* net, Img** imgs, int n) {
    int nCorrect = 0;
    for (int i = 0; i < n; i++) {
        Matrix* prediction = networkPredictImg(net, imgs[i]);
        if (matrixArgmax(prediction) == imgs[i]->label) {
            nCorrect++;
        }
        matrixFree(prediction);
    }
    return 1.0 * nCorrect / n;
}

Matrix* networkPredict(NeuralNetwork* net, Matrix* inputData) {
    Matrix* hiddenInputs = matrixDot(net->hiddenWeights, inputData);
    Matrix* hiddenOutputs = matrixApply(sigmoid, hiddenInputs);
    Matrix* finalInputs = matrixDot(net->outputWeights, hiddenOutputs);
    Matrix* finalOutputs = matrixApply(sigmoid, finalInputs);
    Matrix* result = matrixSoftmax(finalOutputs);

    matrixFree(hiddenInputs);
    matrixFree(hiddenOutputs);
    matrixFree(finalInputs);
    matrixFree(finalOutputs);

    return result;
}

void networkSave(NeuralNetwork* net, char* fileString) {
    mkdir(fileString);
    // Write the descriptor file
    chdir(fileString);
    FILE* descriptor = fopen("descriptor", "w");
    fprintf(descriptor, "%d\n", net->input);
    fprintf(descriptor, "%d\n", net->hidden);
    fprintf(descriptor, "%d\n", net->output);
    fclose(descriptor);
    matrixSave(net->hiddenWeights, "hidden");
    matrixSave(net->outputWeights, "output");
    printf("Successfully written to '%s'\n", fileString);
    chdir("-"); // Go back to the original directory
}

NeuralNetwork* networkLoad(char* fileString) {
    NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
    char entry[MAXCHAR];
    chdir(fileString);

    FILE* descriptor = fopen("descriptor", "r");
    fgets(entry, MAXCHAR, descriptor);
    net->input = atoi(entry);
    fgets(entry, MAXCHAR, descriptor);
    net->hidden = atoi(entry);
    fgets(entry, MAXCHAR, descriptor);
    net->output = atoi(entry);
    fclose(descriptor);
    net->hiddenWeights = matrixLoad("hidden");
    net->outputWeights = matrixLoad("output");
    printf("Successfully loaded network from '%s'\n", fileString);
    chdir("-"); // Go back to the original directory
    return net;
}

void networkPrint(NeuralNetwork* net) {
    printf("# of Inputs: %d\n", net->input);
    printf("# of Hidden: %d\n", net->hidden);
    printf("# of Output: %d\n", net->output);
    printf("Hidden Weights: \n");
    matrixPrint(net->hiddenWeights);
    printf("Output Weights: \n");
    matrixPrint(net->outputWeights);
}

void networkFree(NeuralNetwork* net) {
    matrixFree(net->hiddenWeights);
    matrixFree(net->outputWeights);
    free(net);
    net = NULL;
}
