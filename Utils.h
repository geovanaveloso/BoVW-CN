#ifndef UTILS_H
#define UTILS_H

#include "Group.h"
#include <iostream>
#include <vector>
#include <string>
#include <QVector>
#include <opencv2/opencv.hpp>
#include <Group.h>

using namespace std;
using namespace cv;

class Utils final {
public:
    static QVector<string> getFileNames(const char *directory, QVector<string> names);
    static QVector<string> getFileNamesPascal(string directory, QVector<string> names);
    static bool saveMatrix(const string& filename, const Mat& matrix, const string& matrixname);
    static bool readMatrix(const string& filename, Mat& matrix, const string& matrixname);
    static void printHistogram(Mat histogram);
    static void writeHistograms(Mat histograms, int name, int n, string filename);
    static void writeMapeamentoClasses(string file, string name, int g, int x);
    static float distanciaEuclidiana(Mat p1, Mat p2);
    static Mat convertToWrite(Vec3f v);
    static bool saveBinary(string filename, const Mat& matrix, string matrixname);
    static Mat readBinary(string filename, string matrixname);
    static Mat RGBtoGray(Mat image);
};

#endif
