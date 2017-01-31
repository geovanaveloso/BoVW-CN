#ifndef BOWPROPERTIES_H
#define BOWPROPERTIES_H
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <string.h>

using namespace cv;

#define UNDEFINED -1
#define DEFAULT_CLUSTER_COUNT 30


class BOWProperties {
public:
    // classe em padrão singleton
    static BOWProperties* Instance();
    BOWProperties* setFeatureDetector(string detector);
    Ptr<FeatureDetector> getFeatureDetector();
    BOWProperties* setBOWTrainer(int clusterCount);
    Ptr<BOWKMeansTrainer> getBowTrainer();
    BOWProperties* setDescriptorMatcher(string name);
    Ptr<DescriptorMatcher> getDescriptorMatcher();
    Ptr<BOWImgDescriptorExtractor> getBOWImageDescriptorExtractor();
    BOWProperties* setDescriptorExtractor(string type_descritor);
    BOWProperties* setPathOutput(string outputPath);
    Ptr<DescriptorExtractor> getDescriptorExtractor();
    string getPathOutput();
    void setPathInput(string inputPath);
    string getPathInput();
    void setPathOutResults(string inputPath);
    string getPathOutResults();
    void setTypeVocabularyBuilding(int type);
    void setkNN(int k);
    int getkNN();
    int getTypeVocabularyBuilding();
    string getType_detector();
    string getType_descritor();
    int getFeatureCount();
    int getkCluster();
    void setkCluster(int k);
    void setStoragePath(string path);
    void setLabels(Mat l);
    Mat getLabels();
    void setnCluster(int n);
    int getnCluster();
    string getOutputRoot();
    void setOutputRoot(string outputRoot);
    string getVocPath();
    void setVocPath(string k);
    float getThreshold();
    void setThreshold(float t);
    bool getDatasetColor();
    void setDatasetColor(bool color);
private:
    // classe em padrão singleton
    static BOWProperties* instance;
    Ptr<FeatureDetector> featureDetector;
    Ptr<BOWKMeansTrainer> bowTrainer;
    Ptr<DescriptorMatcher> descriptorMatcher;
    Ptr<BOWImgDescriptorExtractor> bowDE;
    Ptr<DescriptorExtractor> descriptorExtractor;
    int type_vocabulary_building; // 1 para k-means, 2 para OPF não supervisionada e 3 para OPF supervisionado + k-means
    string outputPath;
    string outputRoot;
    string outputResultados;
    string inputPath;
    string vocPath;
    string type_descritor;
    string type_detector;
    int featureCount;
    int k_cluster;
    int k_nn; //default  será 5
    Mat labels;
    int ncluster;
    float t;
    bool dataset_color;
};
#endif // BOWPROPERTIES_H
