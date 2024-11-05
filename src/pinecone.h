/*
Pinecone 0.0.1 - A simple and easy-to-use library for Machine learning
*/


// Includes
#include <stddef.h>




// Definitions

#define MAT_AT(m, i, j) m.es[(i)*(m).cols + (j)]


// Types and structs

typedef struct Matrix
{
   size_t rows;
   size_t cols;
   size_t stride;
   float *es; 
} Matrix;





// Function Declarations


Matrix matrixCreate(size_t rows, size_t cols);
Matrix matrixDot(Matrix m1, Matrix m2);
Matrix matrixSum(Matrix m1, Matrix m2);

void matrixPrint(Matrix matrix);