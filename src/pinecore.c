/* 
Pinecore - Pinecone Core Module.
Contains Core Functionality of the Pinecone Library.
*/



#include "pinecone.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>



Matrix matrixCreate(size_t rows, size_t cols) 
{
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.es = malloc(sizeof(*m.es)*rows*cols);
    assert(m.es != NULL);
    for (size_t i = 0; i < rows * cols; i++) 
    {
        m.es[i] = 0.0f;
    }
    return m;
}


void matrixPrint(Matrix m)
{
    for (size_t i = 0; i < m.rows; i++) 
    {
        for (size_t j = 0; j < m.cols; j++) 
        {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
}