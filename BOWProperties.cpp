#include "BOWProperties.h"

BOWProperties* BOWProperties::instance = NULL;

//Classe com padrão singleton
BOWProperties* BOWProperties::Instance() {
    if (!instance) {
            instance = new BOWProperties;
    }
    return instance;
}

//Set Feature Detector
BOWProperties* BOWProperties::setFeatureDetector(string detector) {
    Ptr<FeatureDetector> featureDetector;
    this->type_detector = detector;
    if (detector.compare("SIFT") == 0){
        featureDetector = new SiftFeatureDetector();
    }else{
        if (detector.compare("SURF") == 0){
            featureDetector = new SurfFeatureDetector();
        }else{
            if (detector.compare("DENSE") == 0){
                featureDetector = new DenseFeatureDetector();
            }else{
                if (detector.compare("FAST") == 0){
                    featureDetector = new FastFeatureDetector();
                }else{
                    if (detector.compare("ORB") == 0){
                        featureDetector = new OrbFeatureDetector();
                    }else{
                        if (detector.compare("STAR") == 0){
                            featureDetector = new StarFeatureDetector();
                        }else{
                            if (detector.compare("MSER") == 0){
                                featureDetector = new MserFeatureDetector();
                            }else{
                                if (detector.compare("BRISK") == 0){
                                    featureDetector = new BRISK(10, 3, 1.0);
                                }else{
                                    if (detector.compare("BLOB") == 0){
                                        featureDetector = new SimpleBlobDetector();
                                    }else{
                                        if (detector.compare("GFTT") == 0){
                                            featureDetector =  new GFTTDetector(1000, 0.01, 1, 3, false, 0.04);
                                        } else{
                                            if (detector.compare("HARRIS") == 0){
                                                featureDetector =  new GFTTDetector(1000, 0.01, 1, 3, true, 0.04);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    this->featureDetector = featureDetector;
    return this;
}

// get FeatureCout (quantos )
int BOWProperties::getFeatureCount() {
   return featureCount;
}

// get k clusters (quantos )
int BOWProperties::getkCluster() {
   return k_cluster;
}

void BOWProperties::setkCluster(int k) {
   k_cluster = k;
}

float BOWProperties::getThreshold() {
   return t;
}

void BOWProperties::setThreshold(float t) {
  this->t = t;
}

string BOWProperties::getVocPath() {
   return vocPath;
}

void BOWProperties::setVocPath(string k) {
   this->vocPath = k;
}

bool BOWProperties::getDatasetColor() {
   return dataset_color;
}

void BOWProperties::setDatasetColor(bool color) {
  this->dataset_color = color;
}

// get k vizinhos (quantos )
int BOWProperties::getkNN() {
   return k_nn;
}

// set k vizinhos (quantos )
void BOWProperties::setkNN(int k) {
    k_nn = k;
}

Mat BOWProperties::getLabels(){
   return labels;
}


void BOWProperties::setLabels(Mat l){
    labels = l;
}


int BOWProperties::getnCluster(){
   return ncluster;
}


void BOWProperties::setnCluster(int n){
    ncluster = n;
}

//Get Feature Detector
Ptr<FeatureDetector> BOWProperties::getFeatureDetector() {
    return featureDetector;
}

// Set k-means trainer
BOWProperties* BOWProperties::setBOWTrainer(int clusterCount){
    this->bowTrainer = new BOWKMeansTrainer(clusterCount);
    this->k_cluster = clusterCount;
    return this;
}

//Get k-means trainer
Ptr<BOWKMeansTrainer> BOWProperties::getBowTrainer(){
    return bowTrainer;
}

//Set descriptor matcher
BOWProperties* BOWProperties::setDescriptorMatcher(string name){
    this->descriptorMatcher = DescriptorMatcher::create(name);
    return this;
}

//Get descriptor matcher
Ptr<DescriptorMatcher> BOWProperties::getDescriptorMatcher(){
    return descriptorMatcher;
}

// Set path onde resultados de treino e classificação vão ser guardados
BOWProperties* BOWProperties::setPathOutput(string storagePath){
    this->outputPath = storagePath;
    return this;
}

// Get path onde resultados de treino e classificação vão ser guardados
string BOWProperties::getPathOutput(){
    return outputPath;
}

// Set path onde resultados de treino e classificação vão ser guardados
void BOWProperties::setPathInput(string storagePath){
    this->inputPath = storagePath;
}

string BOWProperties::getPathOutResults(){
    return outputResultados;
}

void BOWProperties::setPathOutResults(string storagePath){
    this->outputResultados = storagePath;
}

// get path dos descritores e da matriz de distâncias
string BOWProperties::getOutputRoot(){
    return outputRoot;
}


// set path dos descritores e da matriz de distâncias
void BOWProperties::setOutputRoot(string outputRoot){
    this->outputRoot = outputRoot;
}

// Get path onde resultados de treino e classificação vão ser guardados
string BOWProperties::getPathInput(){
    return inputPath;
}


Ptr<BOWImgDescriptorExtractor> BOWProperties::getBOWImageDescriptorExtractor(){
    if (!bowDE){
        bowDE = new BOWImgDescriptorExtractor(descriptorExtractor, descriptorMatcher);
    }
    return bowDE;
}

BOWProperties* BOWProperties::setDescriptorExtractor(string type_descritor) {
    Ptr<DescriptorExtractor> descriptorExtractor;
    this->type_descritor = type_descritor;
    if (type_descritor.compare("SIFT") == 0 || type_descritor.compare("PCA") == 0){
        descriptorExtractor = new SiftDescriptorExtractor();
    } else{
        descriptorExtractor = new SurfDescriptorExtractor();
    }
    this->descriptorExtractor = descriptorExtractor;
    return this;
}

// Get Descriptor Extractor
Ptr<DescriptorExtractor> BOWProperties::getDescriptorExtractor(){
    return descriptorExtractor;
}

// set type of vocabulary building
void BOWProperties::setTypeVocabularyBuilding(int type){
    type_vocabulary_building = type;
}

// get type of vocabulary building
int BOWProperties::getTypeVocabularyBuilding(){
    return type_vocabulary_building;
}

// get type detector
string BOWProperties::getType_detector(){
    return type_detector;
}

string BOWProperties::getType_descritor(){
    return type_descritor;
}
