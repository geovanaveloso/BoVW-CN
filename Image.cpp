#include "Image.h"
#include <map>
#include <bits/stl_map.h>
#include <queue>
#include <algorithm>
#include "Utils.h"
#include "Descriptors.h"

using namespace std;

Image::Image() {
}

Image::Image(string path, bool train){
    this->path = path;
    this->train = train;
    getImage();
}

Image::~Image() {
}

// retorna o caminho
string Image::getImagePath(){
    return this->path;
}

//retorna imagem
void Image::getImage(){
    (this->image.empty()) ? this->image = getImageFromFile() : this->image;
}

bool Image::isTrain(){
    return this->train;
}

//Retorna os keypoints da imagem
void Image::getKeyPoints(){
    keypoints.clear();
    BOWProperties* properties = BOWProperties::Instance();
    if (properties->getType_detector().compare("RANDOM") == 0){
        srand(time(0));
        int n = 1+rand() % 500;
        for (int i=0; i<n; i++){
            srand(time(0));
            int x = 1+rand() % (image.cols-10);
            srand(time(0));
            int y = 1+rand() % (image.rows-10);
            float size;
            srand(time(0));
            (image.cols-x) < (image.rows-y) ? size = 1+rand()%((image.cols)-x) : size = 1+rand()%((image.rows)-y);
            srand(time(0));
            float angle = 1+rand()%360;
            srand(time(0));
            int octave = 1+rand()%4;
            KeyPoint k(x,y, size,angle, -1, octave, 1);
            keypoints.push_back(k);
        }

    }else{
        properties->getFeatureDetector()->detect(Utils::RGBtoGray(this->image), keypoints);
    }
}

//Abre e retorna imagem a partir de um caminho
Mat Image::getImageFromFile(){
    Mat image = imread(this->path, -1);
    return image;
}

//Retorna os descritores da imagem
Mat Image::getDescriptors(){
    this->getKeyPoints();
    BOWProperties* properties = BOWProperties::Instance();

    if (properties->getType_descritor().compare("SIFT") == 0 ||  properties->getType_descritor().compare("SURF") == 0){
        properties->getDescriptorExtractor()->compute(Utils::RGBtoGray(this->image), keypoints, descriptors);
    }else{
        Descriptors d = Descriptors();
        descriptors = d.extractDescriptors(image, keypoints);
    }
    return descriptors;
}

Mat Image::extractHistogram() {
    Descriptors d = Descriptors();
    descriptors = getDescriptors();
    histogram = d.getHistograms(descriptors);
    return histogram;
}

