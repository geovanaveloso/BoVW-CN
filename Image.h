#ifndef IMAGE_H
#define IMAGE_H
#include "BOWProperties.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

class Image{
public:
    Image();
    Image(string path,bool train);
    ~Image();

    string getImagePath();
    void getImage();
    void getKeyPoints();
    Mat getDescriptors();
    Mat extractHistogram();
    bool isTrain();
private:
    Mat image;
    string path;
    bool train;
    vector<KeyPoint> keypoints;
    Mat descriptors;
    Mat histogram;
    Mat getImageFromFile();
};

#endif // IMAGE_H
