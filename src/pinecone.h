/*
Pinecone 0.0.1 - A simple and easy-to-use library for Machine learning
*/






// Types and structs

typedef struct Matrix
{
   int rows;
   int cols;
   int stride;
   float *es; 
} Matrix;





// Function Declarations


Matrix matrix_alloc(int rows, int cols);