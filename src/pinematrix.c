#include "pinecone.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXCHAR 100

Matrix* matrixCreate(int row, int col) {
    Matrix *matrix = malloc(sizeof(Matrix));
    matrix->rows = row;
    matrix->cols = col;
    matrix->entries = malloc(row * sizeof(double*));
    for (int i = 0; i < row; i++) {
        matrix->entries[i] = malloc(col * sizeof(double));
    }
    return matrix;
}

void matrixFill(Matrix *m, int n) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->entries[i][j] = n;
        }
    }
}

void matrixFree(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        free(m->entries[i]);
    }
    free(m->entries);
    free(m);
    m = NULL;
}

void matrixPrint(Matrix* m) {
    printf("Rows: %d Columns: %d\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%1.3f ", m->entries[i][j]);
        }
        printf("\n");
    }
}

Matrix* matrixCopy(Matrix* m) {
    Matrix* mat = matrixCreate(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            mat->entries[i][j] = m->entries[i][j];
        }
    }
    return mat;
}

void matrixSave(Matrix* m, char* file_string) {
    FILE* file = fopen(file_string, "w");
    fprintf(file, "%d\n", m->rows);
    fprintf(file, "%d\n", m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            fprintf(file, "%.6f\n", m->entries[i][j]);
        }
    }
    printf("Successfully saved matrix to %s\n", file_string);
    fclose(file);
}

Matrix* matrixLoad(char* file_string) {
    FILE* file = fopen(file_string, "r");
    char entry[MAXCHAR]; 
    fgets(entry, MAXCHAR, file);
    int rows = atoi(entry);
    fgets(entry, MAXCHAR, file);
    int cols = atoi(entry);
    Matrix* m = matrixCreate(rows, cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            fgets(entry, MAXCHAR, file);
            m->entries[i][j] = strtod(entry, NULL);
        }
    }
    printf("Successfully loaded matrix from %s\n", file_string);
    fclose(file);
    return m;
}

double uniformDistribution(double low, double high) {
    double difference = high - low; // The difference between the two
    int scale = 10000;
    int scaled_difference = (int)(difference * scale);
    return low + (1.0 * (rand() % scaled_difference) / scale);
}

void matrixRandomize(Matrix* m, int n) {
    // Pulling from a random distribution of 
    // Min: -1 / sqrt(n)
    // Max: 1 / sqrt(n)
    double min = -1.0 / sqrt(n);
    double max = 1.0 / sqrt(n);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->entries[i][j] = uniformDistribution(min, max);
        }
    }
}

int matrixArgmax(Matrix* m) {
    // Expects a Mx1 matrix
    double max_score = 0;
    int max_idx = 0;
    for (int i = 0; i < m->rows; i++) {
        if (m->entries[i][0] > max_score) {
            max_score = m->entries[i][0];
            max_idx = i;
        }
    }
    return max_idx;
}

Matrix* matrixFlatten(Matrix* m, int axis) {
    // Axis = 0 -> Column Vector, Axis = 1 -> Row Vector
    Matrix* mat;
    if (axis == 0) {
        mat = matrixCreate(m->rows * m->cols, 1);
    } else if (axis == 1) {
        mat = matrixCreate(1, m->rows * m->cols);
    } else {
        printf("Argument to matrixFlatten must be 0 or 1");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (axis == 0) mat->entries[i * m->cols + j][0] = m->entries[i][j];
            else if (axis == 1) mat->entries[0][i * m->cols + j] = m->entries[i][j];
        }
    }
    return mat;
}

int checkDimensions(Matrix *m1, Matrix *m2) {
    if (m1->rows == m2->rows && m1->cols == m2->cols) return 1;
    return 0;
}

Matrix* matrixMultiply(Matrix *m1, Matrix *m2) {
    if (checkDimensions(m1, m2)) {
        Matrix *m = matrixCreate(m1->rows, m1->cols);
        for (int i = 0; i < m1->rows; i++) {
            for (int j = 0; j < m2->cols; j++) {
                m->entries[i][j] = m1->entries[i][j] * m2->entries[i][j];
            }
        }
        return m;
    } else {
        printf("Dimension mismatch multiply: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
        exit(1);
    }
}

Matrix* matrixAdd(Matrix *m1, Matrix *m2) {
    if (checkDimensions(m1, m2)) {
        Matrix *m = matrixCreate(m1->rows, m1->cols);
        for (int i = 0; i < m1->rows; i++) {
            for (int j = 0; j < m2->cols; j++) {
                m->entries[i][j] = m1->entries[i][j] + m2->entries[i][j];
            }
        }
        return m;
    } else {
        printf("Dimension mismatch add: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
        exit(1);
    }
}

Matrix* matrixSubtract(Matrix *m1, Matrix *m2) {
    if (checkDimensions(m1, m2)) {
        Matrix *m = matrixCreate(m1->rows, m1->cols);
        for (int i = 0; i < m1->rows; i++) {
            for (int j = 0; j < m2->cols; j++) {
                m->entries[i][j] = m1->entries[i][j] - m2->entries[i][j];
            }
        }
        return m;
    } else {
        printf("Dimension mismatch subtract: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
        exit(1);
    }
}

Matrix* matrixApply(double (*func)(double), Matrix* m) {
    Matrix *mat = matrixCopy(m);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            mat->entries[i][j] = (*func)(m->entries[i][j]);
        }
    }
    return mat;
}

Matrix* matrixDot(Matrix *m1, Matrix *m2) {
    if (m1->cols == m2->rows) {
        Matrix *m = matrixCreate(m1->rows, m2->cols);
        for (int i = 0; i < m1->rows; i++) {
            for (int j = 0; j < m2->cols; j++) {
                double sum = 0;
                for (int k = 0; k < m2->rows; k++) {
                    sum += m1->entries[i][k] * m2->entries[k][j];
                }
                m->entries[i][j] = sum;
            }
        }
        return m;
    } else {
        printf("Dimension mismatch dot: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
        exit(1);
    }
}

Matrix* matrixScale(double n, Matrix* m) {
    Matrix* mat = matrixCopy(m);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            mat->entries[i][j] *= n;
        }
    }
    return mat;
}

Matrix* matrixAddScalar(double n, Matrix* m) {
    Matrix* mat = matrixCopy(m);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            mat->entries[i][j] += n;
        }
    }
    return mat;
}

Matrix* matrixTranspose(Matrix* m) {
    Matrix* mat = matrixCreate(m->cols, m->rows);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            mat->entries[j][i] = m->entries[i][j];
        }
    }
    return mat;
}
