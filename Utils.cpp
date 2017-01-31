#include "Utils.h"
#include <omp.h>
#include <QVector>
#include <dirent.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <fstream>
#include "Dataset.h"
#include <math.h>
#include <Group.h>
#include <fstream>

using namespace cv;
using namespace std;

QVector<string> Utils::getFileNames(const char *directory, QVector<string> names) {
    names.clear();
    DIR *dir = 0;
    struct dirent *pasta = 0;
    unsigned char isDir = 0x4;
    dir = opendir (directory);
    string fileName;

    if (dir != 0) {
        while (pasta = (readdir (dir))){
            fileName = pasta->d_name;
            if (!fileName.compare(".") == 0 && !fileName.compare("..") == 0){
                if (fileName.find(".jpg") != -1 ||  fileName.find(".png") != -1 ||  fileName.find(".tiff") != -1 ||  fileName.find(".tif") != -1  || fileName.find(".bmp") != -1 ||
                        fileName.find(".BMP") != -1 ||pasta->d_type == isDir)
                names.push_back(fileName);
            }
        }
        closedir (dir);
        return names;

    }else{
        exit(1);
    }
}

bool Utils::saveBinary(string filename, const Mat& matrix, string matrixname){
    std::setlocale(LC_ALL, "en_US.UTF-8");
    FILE *fpOut = NULL;
    filename.append(matrixname);
    string nametxt = filename+".txt";
    ofstream out;
    out.open(nametxt.c_str(), ios::app);
    float f = 0.0;
    fpOut = fopen(filename.c_str(),"wb");
    fwrite(&matrix.cols,sizeof(int),1,fpOut);
    out << matrix.cols << " ";
    fwrite(&matrix.rows,sizeof(int),1,fpOut);
    out << matrix.rows << endl;
    for (int i =0; i<matrix.rows; i++){
        for (int j=0; j<matrix.cols; j++){
            fwrite(&matrix.at<float>(i,j),sizeof(float),1,fpOut);
            f = matrix.at<float>(i,j);
            out << f << " ";
        }
        out << endl;
     }
    fclose(fpOut);
}

Mat Utils::readBinary(string filename, string matrixname){
    Mat binary, b_aux;
    FILE *fpIn = NULL;
    int cols, rows;
    float aux;
    filename.append(matrixname);
    fpIn = fopen(filename.c_str(),"rb");
    if (fpIn != NULL){
        fread(&cols,sizeof(int),1,fpIn);
        fread(&rows,sizeof(int),1,fpIn);
        for (int i=0; i<rows; i++){
            for (int j=0; j<cols; j++){
                fread(&aux,sizeof(float),1,fpIn);
                b_aux.push_back(aux);
            }
            binary.push_back(b_aux.reshape(b_aux.channels(), 1));
            b_aux.release();
        }
        fclose(fpIn);
    }
    return binary;
}

bool Utils::saveMatrix(const string& filename, const Mat& matrix, const string& matrixname){
    FileStorage fs(filename+matrixname+".xml", FileStorage::APPEND);
    if (fs.isOpened()) {
        fs << matrixname << matrix;
        fs.release();
        return true;
    }
    fs.release();
    return false;
}

bool Utils::readMatrix(const string& filename, Mat& matrix, const string& matrixname){
    FileStorage fs(filename+matrixname+".xml", FileStorage::READ);
    if (fs.isOpened())	{
        fs[matrixname] >> matrix;
        fs.release();
        return !matrix.empty();
    }
    return false;
    fs.release();

}

Mat Utils::convertToWrite(Vec3f v){
   Mat mat_lab(1, 3, CV_32F);
    mat_lab.at<float>(0) = v.val[0];
    mat_lab.at<float>(1) = v.val[1];
    mat_lab.at<float>(2) = v.val[2];
    return mat_lab;
}

void Utils::writeHistograms(Mat histograms, int name, int n, string filename) {
    ofstream outOPF;
    outOPF.open(filename.c_str(), ios::app);
    outOPF << n << " " << name << " ";

    for (int i = 0; i < histograms.rows; i++)	{
        for (int j = 0; j < histograms.cols; j++){
            outOPF << histograms.at<float>(i, j) << " ";
        }
    }
    outOPF << endl;
}

void Utils::writeMapeamentoClasses(string filename, string name, int g, int x){
    ofstream out;
    filename.append("Mapeamento_classes");
    out.open(filename.c_str(), ios::app);
    out << g << " = " << name << "=" << x << endl;
}


float Utils::distanciaEuclidiana(Mat p1, Mat p2){
    float result = 0, x1, x2;
    for (int i =0; i<p1.cols; i++){
        x1 = p1.at<float>(0, i);
        x2 = p2.at<float>(0, i);
        result+= pow((x1-x2),2);
    }
    return sqrt(result);
}

Mat Utils::RGBtoGray(Mat image){
    if (image.channels()!=1){
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    }
    return image;
}

