#include "pinecone.h"
#include <math.h>

double sigmoid(double input) {
    return 1.0 / (1 + exp(-input));
}

Matrix* matrixSigmoidPrime(Matrix* m) {
    Matrix* ones = matrixCreate(m->rows, m->cols);
    matrixFill(ones, 1);
    Matrix* subtracted = matrixSubtract(ones, m);
    Matrix* multiplied = matrixMultiply(m, subtracted);
    matrixFree(ones);
    matrixFree(subtracted);
    return multiplied;
}

Matrix* matrixSoftmax(Matrix* m) {
    double total = 0;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            total += exp(m->entries[i][j]);
        }
    }
    Matrix* mat = matrixCreate(m->rows, m->cols);
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            mat->entries[i][j] = exp(m->entries[i][j]) / total;
        }
    }
    return mat;
}
