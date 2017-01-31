#ifndef GROUP_H
#define GROUP_H

#include "Image.h"
#include "BOWProperties.h"
#include "Utils.h"

using namespace cv;

class Group
{
public:
    ~Group();
    Group(string path, string name);

    string getPath();
    string getName();
    vector<Image> images;


    void trainBOW();
    int getHistograms(int n, int g, string filename);
    void trainGroupClassifier();

private:
    string path;
    string name;
    Mat groupClasifier;
};

#endif
