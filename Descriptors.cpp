#include "Descriptors.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "Utils.h"
#include <map>
#include "Graph.h"
#include <igraph/igraph.h>

std::ostringstream sw;

struct GMatch {
  int   queryIdx;   //index query
  int   trainIdx;    //index train
  float distance;      // distancia
};

bool mysort2 (GMatch lhs, GMatch rhs) { return lhs.distance < rhs.distance; }

Descriptors::Descriptors(){
}

Descriptors::~Descriptors() {
}

Mat Descriptors::extractAreaKeypoint(Mat image, KeyPoint k){
    float radius = (ceil(k.size)/2)+2;
    int x, y, w, h;
    x = cvRound(k.pt.x-(radius));
    y = cvRound(k.pt.y-(radius));
    w = cvRound(k.pt.x+(radius)) - x;
    h = cvRound(k.pt.y+(radius)) - y;

    if (x<0) x=0;
    if (x>image.cols) x=image.cols;
    if (y<0) y=0;
    if (y>image.rows) y=image.rows;
    if (w<0) w=0;
    if ((w+x)>image.cols) w= (image.cols-x);
    if (h<0) h=0;
    if ((h+y)>image.rows) h=(image.rows-y);

    Mat img_recorte(image, Rect(x,y,w,h));
    return img_recorte;
}

Mat Descriptors::extractDescriptors(Mat image, vector<KeyPoint> keypoints){
    Mat descritor,res_final;
    BOWProperties* properties = BOWProperties::Instance();

    for (int i=0; i<(int)keypoints.size(); i++){
        res_final =  extractAreaKeypoint(image, keypoints[i]);
        if (properties->getType_descritor().compare("BOC") == 0) {
            cv::cvtColor(res_final, res_final, cv::COLOR_BGR2Lab);
            res_final.convertTo(res_final, CV_32FC3);
            descritor.push_back(Utils::convertToWrite(extractBOC(res_final)));
        }else{
            if (properties->getType_descritor().compare("LBP") == 0) {
                descritor.push_back(extractLBP(Utils::RGBtoGray(res_final)));
            }else{
                if (properties->getType_descritor().compare("BIC") == 0) {
                    descritor.push_back(extractBIC(res_final));
                }else{
                    if (properties->getType_descritor().compare("Fourier")==0){
                        descritor.push_back(Fourier(Utils::RGBtoGray(res_final)));
                    }else{
                        if (properties->getType_descritor().compare("Redes")==0){
                            descritor.push_back(extractNetworks(Utils::RGBtoGray(res_final), properties->getThreshold()));
                        }else{
                             descritor.push_back(extractNetworks1(Utils::RGBtoGray(res_final)));
                        }
                    }
                }
            }
        }
        res_final.release();
    }

    //    if (pca){
    //        normalize(descritor, descritor, 0, 1, NORM_MINMAX, -1, Mat());
    //        Mat features = descritor; descritor.release();
    //        PCA pca(features,Mat(),CV_PCA_DATA_AS_ROW,20);
    //        pca.project(features, descritor);
    //        if (descritor.cols!=36){
    //            Mat col(descritor.rows, (36-descritor.cols), CV_32F);
    //            col = Mat::zeros(descritor.rows, (36-descritor.cols), CV_32F);
    //            hconcat(descritor, col, descritor);
    //        }
    //    }
    descritor.convertTo(descritor,CV_32F);
    return descritor;
}

Mat Descriptors::extractPCASIFT(Mat img){
    Mat grad_x, grad_y;
    Sobel(img, grad_x, CV_32FC1, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    Sobel(img, grad_y, CV_32FC1 , 0, 1, 3, 1, 0, BORDER_DEFAULT);
    grad_x.push_back(grad_y);
    if (grad_x.cols!=39){
        Mat col(grad_x.rows, (39-grad_x.cols), CV_32F);
        col = Mat::zeros(grad_x.rows, (39-grad_x.cols), CV_32F);
        hconcat(grad_x, col, grad_x);
    }
    if (grad_x.rows!=78){
        if ((78-grad_x.rows)<0){
                grad_x.pop_back(grad_x.rows-78);
        }else{
            Mat col((78-grad_x.rows), grad_x.cols, CV_32F);
            col = Mat::zeros((78-grad_x.rows), grad_x.cols, CV_32F);
            vconcat(grad_x, col, grad_x);
        }
    }
    grad_x = grad_x.reshape(1,1);
    return grad_x;
}

class Vec3fWrap {
public:
    Vec3fWrap(Vec3f v) : vec_(v){}
    bool operator<(const Vec3fWrap& v1) const {
        return (vec_[0] < v1.vec()[0]) && (vec_[1] < v1.vec()[1]) && (vec_[2] < v1.vec()[2]);
    }
    Vec3f vec() const { return vec_; }
private:
     Vec3f vec_;
};

Vec3f Descriptors::extractBOC(Mat img) {
    std::map<Vec3fWrap, int> colors;
    int max_count = 0;
    Vec3f common_color;

    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {
            common_color = img.at<Vec3f>(row,col);

            Vec3fWrap Lab(common_color);
            if (colors[Lab]++ > max_count) {
                common_color = Lab.vec();
                max_count = colors.at(Lab);
            }
        }
    }
    return common_color;
}

int Descriptors::getValueCenter(int x, int y, Mat src) {
    uchar vizinho[8];
    uchar centro;
    int r = 0;
    centro = src.at<uchar>(y, x);
    vizinho[0] = src.at<uchar>(y-1, x-1);
    vizinho[1] = src.at<uchar>(y-1, x);
    vizinho[2] = src.at<uchar>(y-1, x+1);
    vizinho[3] = src.at<uchar>(y, x-1);
    vizinho[4] = src.at<uchar>(y, x+1);
    vizinho[5] = src.at<uchar>(y+1, x-1);
    vizinho[6] = src.at<uchar>(y+1, x);
    vizinho[7] = src.at<uchar>(y+1, x+1);

    if (vizinho[0] >= centro) r += 1;
    if (vizinho[1] >= centro) r += 2;
    if (vizinho[2] >= centro) r += 4;
    if (vizinho[4] >= centro) r += 8;
    if (vizinho[7] >= centro) r += 16;
    if (vizinho[6] >= centro) r += 32;
    if (vizinho[5] >= centro) r += 64;
    if (vizinho[3] >= centro) r += 128;

    return r;
}

Mat Descriptors::extractLBP(Mat image) {
    vector<float> resp;
    int histSize = 256;
    float vet[histSize];

    for(int i = 0; i < histSize; i++) {
        vet[i] = 0;
    }
    Size imgSize = image.size();
    for(int y = 1; y < imgSize.height-1; y ++) {
        for(int x = 1; x < imgSize.width-1; x++) {
            vet[getValueCenter(x,y,image)]++;
        }
    }
    for (int i = 0; i < histSize; ++i) {
        resp.push_back(vet[i]);
    }

    Mat hist(resp, true);
    hist = hist.reshape(1, 1);
    return hist;
}

Mat Descriptors::getHistograms(Mat descritores){
    BOWProperties* properties = BOWProperties::Instance();
    Mat vocabulary, hist, labels, d;
    vocabulary = properties->getBOWImageDescriptorExtractor()->getVocabulary();
    labels = properties->getLabels();
    hist = Mat::zeros(1, properties->getkCluster(), CV_32F);
    std::vector<GMatch> matches;
    int n;
    GMatch de;

    for (int i =0; i<descritores.rows; i++){
        de.queryIdx = i;
        for (int j=0; j<vocabulary.rows; j++){
            de.trainIdx =  j;
            de.distance = Utils::distanciaEuclidiana(descritores.row(i), vocabulary.row(j));
            matches.push_back(de);
        }
        std::sort(matches.begin(), matches.end(), mysort2);
        if (properties->getTypeVocabularyBuilding() <= 2){
            ++(hist.at<float>(matches[0].trainIdx));
        }else{
            n = labels.at<int>(matches[0].trainIdx, 0);
            ++(hist.at<float>(n));
        }
        matches.clear();
        d.release();
    }
    vocabulary.release();
    labels.release();
    return hist;
}

Mat Descriptors::extractBIC(Mat image) {
    vector<int> hist;
    for(int i =0; i<128;i++){
        hist.push_back(0);
    }
    Mat requantizedImage;
    vector<uchar> interior;
    vector<uchar> border;
    Vec<int, 64> histogramInterior;
    Vec<int, 64> histogramBorder;
    requantizedImage = requantize(image);
    classify(requantizedImage, &interior, &border);
    computeHistogram(interior, &histogramInterior);
    computeHistogram(border, &histogramBorder);
    vector<int> h = concateneHistogram(histogramInterior, histogramBorder);
    Mat hi(h, true);
    hi = hi.reshape(1, 1);
    hi.convertTo(hi, CV_32FC1);
    return hi;
}

Mat Descriptors::requantize(Mat image) {
    Mat requantizedImage = Mat::zeros(image.rows, image.cols, CV_8U);
    Size size = image.size();
    for (int i = 0; i < size.height; i++) {
        for (int j = 0; j < size.width; j++) {
            Vec3b pixel = image.at<Vec3b>(i, j);
            uchar r = ((pixel.val[0] & 0b11000000) >> 2);
            uchar g = ((pixel.val[1] & 0b11000000) >> 4);
            uchar b = ((pixel.val[2] & 0b11000000) >> 6);
            uchar rgb = r | g | b;
            requantizedImage.at<uchar>(i, j) = rgb;
        }
    }
    return requantizedImage;
}

vector<int> Descriptors::concateneHistogram(Vec<int, 64> histogram1, Vec<int, 64> histogram2) {
    vector<int> res;
    for (int i = 0; i < 64; i++) {
        res.push_back(histogram1[i]);
    }
    for (int i = 0; i < 64; i++) {
        res.push_back(histogram2[i]);
    }
    return res;
}

void Descriptors::computeHistogram(vector<uchar> pixels, Vec<int, 64> *histogram) {
    for (vector<uchar>::iterator it = pixels.begin(); it != pixels.end(); ++it) {
        (*histogram)[*it] = (*histogram)[*it] + 1;
    }
}

void Descriptors::classify(Mat image, vector<uchar> *interior, vector<uchar> *border) {
    Size size = image.size();
    for (int i = 0; i < size.height; i++) {
        for (int j = 0; j < size.width; j++) {
            if (isInterior(image,  j, i)) {
                (*interior).push_back(image.at<uchar>(i, j));
            } else {
                (*border).push_back(image.at<uchar>(i, j));
            }
        }
    }
}

bool Descriptors::isInterior(Mat image, int x, int y) {
    if (y == 0 || x == 0 || y == image.rows - 1 || x == image.cols - 1) {
        return false;
    }
    if (image.at<uchar>(y, x) != image.at<uchar>(y - 1, x)) {
        return false;
    }
    if (image.at<uchar>(y, x) != image.at<uchar>(y + 1, x)) {
        return false;
    }
    if (image.at<uchar>(y, x) != image.at<uchar>(y, x - 1)) {
        return false;
    }
    if (image.at<uchar>(y, x) != image.at<uchar>(y, x + 1)) {
        return false;
    }
    return true;
}

Mat Descriptors::extractNetworks(Mat image, float t){
    vector<Graph> graphs;
    Mat desc;
    Graph graphInicial, g;
    graphInicial.addVertex(image);

    if (t==0){
        for (float i=0.005; i<=0.53; i+=0.015){
            g = graphInicial.thresholds(graphInicial, i);
            graphs.push_back(g);
            g.release();
        }

        for (int i=0; i<(int)graphs.size(); i++){
            desc.push_back(graphInicial.extractFeatures(graphs.at(i)));
        }
    }else{
        g = graphInicial.thresholds(graphInicial, t);
        desc.push_back(graphInicial.extractFeatures(g));
    }
    desc = desc.reshape(desc.channels(), 1);
    return desc;
}

Mat Descriptors::extractNetworks1(Mat image){
    vector<Graph> graphs;
    Mat desc;
    Graph graphInicial, g;
    graphInicial.addVertex(image);

    for (float i=0.005; i<=0.53; i+=0.015){
        g = graphInicial.thresholds(graphInicial, i);
        graphs.push_back(g);
        g.release();
    }

    for (int i=0; i<(int)graphs.size(); i++){
        desc.push_back(graphInicial.extractFeatures1(graphs.at(i)));
    }

    desc = desc.reshape(desc.channels(), 1);
    return desc;
}

Mat Descriptors::Fourier(Mat image){
    Mat padded;
    CvPoint2D32f center;
    int m = getOptimalDFTSize( image.rows );
    int n = getOptimalDFTSize( image.cols );
    copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI);
    split(complexI, planes);
    magnitude(planes[0], planes[1], planes[0]);
    Mat magI = planes[0];
    magI += Scalar::all(1);
    log(magI, magI);
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    center.x = magI.cols/2;
    center.y = magI.rows/2;
    Mat q0(magI, Rect(0, 0, center.x, center.y));
    Mat q1(magI, Rect(center.x, 0, center.x, center.y));
    Mat q2(magI, Rect(0, center.y, center.x, center.y));
    Mat q3(magI, Rect(center.x, center.y, center.x, center.y));
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
    normalize(magI, magI, 0, 1, CV_MINMAX);
    float r=5.625, radius=8;
    int sum;
    Mat desc;

    for (float i=0; i<360; i+=r){
        Mat mask(magI.size(), CV_8UC1, Scalar(255,255,255));
        ellipse(mask, center, Size(radius, radius), 0, i, (i+r), Scalar( 0, 0, 0 ), -1, 8);
         bitwise_not(mask, mask);
         Mat masked;
         magI.copyTo(masked, mask);

         sum=0;
         for (int x=0; x<masked.rows; x++){
             for(int y=0; y<masked.cols; y++){
               sum+=masked.at<int>(x,y);
             }
         }
         desc.push_back(sum);
    }
    desc = desc.reshape(desc.channels(), 1);
    return desc;
}


