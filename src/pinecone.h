/*
Pinecone - Easy to use Machine Learning Library
Made by SusgUY446
License: GPL 3.0
*/

typedef struct {
    double** entries;
    int rows;
    int cols;
} Matrix;

Matrix* matrixCreate(int row, int col);
void matrixFill(Matrix *m, int n);
void matrixFree(Matrix *m);
void matrixPrint(Matrix *m);
Matrix* matrixCopy(Matrix *m);
void matrixSave(Matrix* m, char* fileString);
Matrix* matrixLoad(char* fileString);
void matrixRandomize(Matrix* m, int n);
int matrixArgmax(Matrix* m);
Matrix* matrixFlatten(Matrix* m, int axis);

Matrix* matrixMultiply(Matrix* m1, Matrix* m2);
Matrix* matrixAdd(Matrix* m1, Matrix* m2);
Matrix* matrixSubtract(Matrix* m1, Matrix* m2);
Matrix* matrixDot(Matrix* m1, Matrix* m2);
Matrix* matrixApply(double (*func)(double), Matrix* m);
Matrix* matrixScale(double n, Matrix* m);
Matrix* matrixAddScalar(double n, Matrix* m);
Matrix* matrixTranspose(Matrix* m);

typedef struct {
    Matrix* imgData;
    int label;
} Img;

Img** csvToImgs(char* fileString, int numberOfImgs);
void imgPrint(Img* img);
void imgFree(Img *img);
void imgsFree(Img **imgs, int n);

typedef struct {
    int input;
    int hidden;
    int output;
    double learningRate;
    Matrix* hiddenWeights;
    Matrix* outputWeights;
} NeuralNetwork;

NeuralNetwork* networkCreate(int input, int hidden, int output, double lr);
void networkTrain(NeuralNetwork* net, Matrix* inputData, Matrix* outputData);
void networkTrainBatchImgs(NeuralNetwork* net, Img** imgs, int batchSize);
Matrix* networkPredictImg(NeuralNetwork* net, Img* img);
double networkPredictImgs(NeuralNetwork* net, Img** imgs, int n);
Matrix* networkPredict(NeuralNetwork* net, Matrix* inputData);
void networkSave(NeuralNetwork* net, char* fileString);
NeuralNetwork* networkLoad(char* fileString);
void networkPrint(NeuralNetwork* net);
void networkFree(NeuralNetwork* net);

double sigmoid(double input);
Matrix* matrixSigmoidPrime(Matrix* m);
Matrix* matrixSoftmax(Matrix* m);