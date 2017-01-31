#ifndef DESCRIPTORS_H
#define DESCRIPTORS_H
#include "BOWProperties.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <igraph/igraph.h>

class Descriptors
{   
public:
    Descriptors();
    ~Descriptors();

    Mat extractAreaKeypoint(Mat image,KeyPoint k);
    Vec3f extractBOC(Mat img);
    Mat extractLBP(Mat image);
    int getValueCenter(int x, int y, Mat src);
    Mat extractLabHistogram(Mat image, vector<KeyPoint> keypoints);
    Mat getHistograms(Mat descritores);
    Mat extractDescriptors(Mat image, vector<KeyPoint> keypoints);
    Mat extractPCASIFT(Mat img);
    Mat requantize(Mat image);
    Mat extractBIC(Mat image);
    void classify(Mat image, vector<uchar> *interior, vector<uchar> *border);
    bool isInterior(Mat image, int x, int y);
    void computeHistogram(vector<uchar> pixels, Vec<int, 64> *histogram);
    vector<int> concateneHistogram(Vec<int, 64> histogram1, Vec<int, 64> histogram2);
    Mat extractNetworks(Mat image, float t);
    Mat extractNetworks1(Mat image);
    Mat Fourier(Mat image);
};

#endif // DESCRIPTORS_H
