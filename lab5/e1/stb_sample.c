#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <stdio.h>
#include <stdlib.h>

typedef unsigned char uchar;

uchar *load_jpeg_as_rgb(char *filename, int *width, int *height)

// efekt:   wczytanie pliku jpeg
// wynik:   wskaźnik na przydzielony bufor, w którym wiersz za wierszem
//          umieszczona jest reprezentacja bitmapy, każdy piksel zajmuje
//          3 bajty (R, G, B); NULL w przypadku błędu wczytywania;
//          otrzymany bufor trzeba zwolnić przez stb_image_free()
// wejście: filename - nazwa pliku jpeg
// wyjście: width    - szerokosć obrazu (piksele)
//          height   - wysokość obrazu (piksele)
{
    int channels;
    uchar *data = stbi_load(filename, width, height, &channels, 3);
    if (!data) {
        fprintf(stderr, "stbi_load failed: %s\n", stbi_failure_reason());
        return NULL;
    }
    return data;
}

int main(int argc, char *argv[]){
    if(argc != 2) { 
        fprintf(stderr, "usage: %s jpg_file\n", argv[0]); 
        return 1; 
    }
    int w, h;
    uchar *pixels = load_jpeg_as_rgb(argv[1], &w, &h);
    if(!pixels) 
        return 1;
    printf("Dimensions: %d x %d\n", w, h);

    int x = 0, y = 0;
    unsigned char r = pixels[(y*w + x)*3 + 0];
    unsigned char g = pixels[(y*w + x)*3 + 1];
    unsigned char b = pixels[(y*w + x)*3 + 2];
    printf("Pixel[0,0] = R=0x%X G=0x%X B=0x%X\n", r, g, b);

    stbi_image_free(pixels);
    return 0;
}

