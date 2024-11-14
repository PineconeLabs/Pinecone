#include "pinecone.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXCHAR 10000

Img** csvToImgs(char* fileString, int numberOfImgs) {
    FILE *fp;
    Img** imgs = malloc(numberOfImgs * sizeof(Img*));
    char row[MAXCHAR];
    fp = fopen(fileString, "r");

    // Read the first line 
    fgets(row, MAXCHAR, fp);
    int i = 0;
    while (feof(fp) != 1 && i < numberOfImgs) {
        imgs[i] = malloc(sizeof(Img));

        int j = 0;
        fgets(row, MAXCHAR, fp);
        char* token = strtok(row, ",");
        imgs[i]->imgData = matrixCreate(28, 28); // Updated to matrixCreate
        while (token != NULL) {
            if (j == 0) {
                imgs[i]->label = atoi(token);
            } else {
                imgs[i]->imgData->entries[(j-1) / 28][(j-1) % 28] = atoi(token) / 256.0;
            }
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }
    fclose(fp);
    return imgs;
}

void imgPrint(Img* img) {
    matrixPrint(img->imgData);  // Updated to matrixPrint
    printf("Img Label: %d\n", img->label);
}

void imgFree(Img* img) {
    matrixFree(img->imgData);  // Updated to matrixFree
    free(img);
    img = NULL;
}

void imgsFree(Img** imgs, int n) {
    for (int i = 0; i < n; i++) {
        imgFree(imgs[i]);  // Updated to imgFree
    }
    free(imgs);
    imgs = NULL;
}
