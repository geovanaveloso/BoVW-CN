#include "Dataset.h"
#include "Image.h"
#include "Classifier.h"
#include "Descriptors.h"
#include <algorithm>
#include <fstream>
#include <QVector>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <omp.h>
#include "opencv2/ocl/ocl.hpp"
#include <ctime>
#include <sys/stat.h>
extern "C" {
#include "OPF.h"
}

struct GMatch {
    int   queryIdx;   //index query
    int   trainIdx;    //index train
    float distance;      // distancia
};

bool mysort (GMatch lhs, GMatch rhs) { return lhs.distance < rhs.distance; }
bool mysortinv (GMatch lhs, GMatch rhs) { return lhs.distance > rhs.distance; }

using namespace std;
std::ostringstream st;

Dataset::Dataset(string path) {
    this->path = path;
    QVector<string> classNames;
    classNames = Utils::getFileNames(this->path.c_str(), classNames);
    foreach (string className, classNames){
        Group group(path + className, className);
        groups.push_back(group);
    }
}

Dataset::~Dataset(){
}

void Dataset::trainClassifier(){
    BOWProperties* properties = BOWProperties::Instance();
    string filename = properties->getPathOutput()+"histograms-original";
    writeCabecalhoOPF(filename, properties->getkCluster());
    int n = 0, g;
    cout << "Creating histograms" << endl;
    for (g=0; g<(int)groups.size(); g++) {
        cout << g << endl;
        n = groups[g].getHistograms(n,g, filename);       
    }
    cout << "Original histograms computed" << endl;
}

void Dataset::classifier(){
    Classifier opf = Classifier();
    opf.classifier();
}

void Dataset::trainBOW(){
    BOWProperties* properties = BOWProperties::Instance();
    Mat vocabulary, descritores, labels, l;
    std::cout << " Training BOW "  << std::endl;
    bool quant = false; int f=1;
    if (!Utils::readMatrix(properties->getVocPath(), vocabulary, "vocabulary")){
        descritores = Utils::readBinary(properties->getOutputRoot(), "descritores");
        if (descritores.empty()){
            foreach (Group group, groups){
                std::cout << f << std::endl;
                group.trainBOW();
                f++;
            }
            vector <Mat> desc = properties->getBowTrainer()->getDescriptors();
            for (int j=0; j<(int)desc.size(); j++){
                descritores.push_back(desc[j]);
            }
            Mat d_teste = Utils::readBinary(properties->getOutputRoot(), "descritores");
            if (d_teste.empty()){
                Utils::saveBinary(properties->getOutputRoot(), descritores, "descritores");
            }
        }
        if (Utils::readMatrix(properties->getPathInput(), vocabulary, "centroides")){
            quant = true;
            Utils::readMatrix(properties->getPathInput(), labels, "labels");
            Utils::readMatrix(properties->getVocPath(), l, "labels-vocabulary");
            if (l.empty()){ for (int i=0; i<vocabulary.rows; i++){ l.push_back(i);}}
        }
            switch (properties->getTypeVocabularyBuilding()) {
            case 0:{
                cout << "Running trainer with random " << endl;
                if (!quant){
                    vocabulary = trainerRandom(descritores);
                }
                break;
            }
            case 1: {
                cout << "Running trainer with k-means" << endl;
                if (!quant){
                    vocabulary = trainerKmeans(descritores);
                }
                break;
            }
            case 2 : {
                cout << "Running trainer with OPF unsupervised" << endl;
                if (!quant){
                    vocabulary = trainerOPF(descritores);
                }
                break;
            }
            case 4:{
                cout << "Running trainer with borders" << endl;
                if (!quant){
                    vocabulary = trainerKmeans(descritores);
                    Utils::readMatrix(properties->getPathInput(), labels, "labels");
                }
                vocabulary.release();
                l.release();
            }
            case 3:{
                vocabulary = bordasPorClusters(vocabulary, descritores, l, labels);
                break;
            }
            case 6:{
                cout << "Running trainer with borders" << endl;
                if (!quant){
                    vocabulary = trainerKmeans(descritores);
                    Utils::readMatrix(properties->getPathInput(), labels, "labels");
                }
                vocabulary.release();
                l.release();
            }
            case 5:{
                cout << "Running trainer with borders" << endl;
                vocabulary = bordasPorClusters(vocabulary, descritores, l, labels);
                break;
            }
            case 7:{
                if (!quant){
                    vocabulary = trainerKmeans(descritores);
                    Utils::readMatrix(properties->getPathInput(), labels, "labels");
                    Utils::readMatrix(properties->getVocPath(), l, "labels-vocabulary");
                    if (l.empty()){ for (int i=0; i<vocabulary.rows; i++){ l.push_back(i);}}
                }
                vocabulary = rotuloDistintoMaisProxPorCluster(vocabulary, descritores, l, labels);
                break;
            }
            }
        Utils::saveMatrix(properties->getVocPath(), vocabulary, "vocabulary");
    }
    properties->setLabels(l);
    properties->setkCluster(vocabulary.rows);
    properties->getBOWImageDescriptorExtractor()->setVocabulary(vocabulary);
    Utils::readMatrix(properties->getVocPath(), l, "labels-vocabulary");

}

Mat Dataset::trainerKmeans(Mat descritores){
    BOWProperties* properties = BOWProperties::Instance();
    Mat vocabulary, labels;
    kmeans(descritores,properties->getkCluster(),labels,TermCriteria(), 5, KMEANS_PP_CENTERS, vocabulary);
    Utils::saveMatrix(properties->getPathInput(), labels, "labels");
    Utils::saveMatrix(properties->getPathInput(), vocabulary, "centroides");
    return vocabulary;
}

void Dataset::writeCabecalhoOPF(string filename, int n_caracteristicas){
    int n_classes = 0, n_dados=0;
    foreach (Group group, groups) {
        n_classes++;
        for (int i =0; i<(int)group.images.size(); i++){
            if (!group.images[i].isTrain()){
                n_dados++;
            }
        }
    }
    ofstream out;
    out.open(filename.c_str(), ios::app);
    out << n_dados << " " << n_classes << " " << n_caracteristicas << endl;
}

void Dataset::writeCabecalhoARFF(string filename){
    int k =  BOWProperties::Instance()->getkCluster();
    string name_full = "./"+filename+".txt";
    ofstream out;
    out.open(name_full.c_str(), ios::app);
    
    out << "@RELATION " << filename << endl;
    
    for (int i =0; i<k; i++){
        out << "@ATTRIBUTE a"<<i<< " REAL" << endl;
    }
    out << "@ATTRIBUTE class {";
    for (int g=0; g<(int)groups.size(); g++) {
        out << g ;
        if (g != (int) (groups.size() - 1)){
            out << ", ";
        }
    }
    out << " }" << endl;
    
    out << "@DATA" << endl;
}

Mat Dataset::trainerRandom(Mat descritores){
    Mat vocabulary;
    int x, i=0, desc_tamanho = descritores.rows;
    BOWProperties* properties = BOWProperties::Instance();
    while(i<properties->getkCluster()){
        srand(time(0));
        x = rand()%desc_tamanho;
        vocabulary.push_back(descritores.row(x));
        i++;
    }
    return vocabulary;
}

Mat Dataset::trainerOPF(Mat descritores){
    int n=0;
    ofstream out;
    BOWProperties* properties = BOWProperties::Instance();
    Mat vocabulary;
    string name_full = properties->getPathOutput()+"vocabulary";
    out.open(name_full.c_str(), ios::app);
    out << descritores.rows << " 1 " << descritores.cols<< endl;
    for (int i=0; i<descritores.rows; i++){
        out << n << " 0 ";
        for (int j=0; j<descritores.cols; j++){
            out << descritores.at<float>(i,j) << " ";
        }
        out << endl;
        n++;
    }
    Classifier opf = Classifier();
    vector <int> roots = opf.vocabularyWithOPF("vocabulary");
    for (int i =0; i<(int)roots.size(); i++){
        vocabulary.push_back(descritores.row(roots.at(i)));
    }
    properties->setkCluster(roots.size());
    Utils::saveMatrix(properties->getPathInput(), vocabulary, "centroides");
    return vocabulary;
}

// Encontrar bordas dos cluster
// ver se entre os k vizinhos tem amostra com rótulo distinto
Mat Dataset::bordasPorClusters(Mat vocabulary, Mat descritores, Mat l, Mat labels){
    GMatch de;
    vector<GMatch> matches;
    BOWProperties* properties = BOWProperties::Instance();

    for (int i=0; i<descritores.rows; i++){
        de.queryIdx = i;
        for (int j=0; j<descritores.rows; j++){
            if (i<j || i!=j){
                de.trainIdx =  j;
                de.distance = Utils::distanciaEuclidiana(descritores.row(i), descritores.row(j));
                matches.push_back(de);
            }
        }
        std::sort(matches.begin(), matches.end(), mysort);
        for(int j=0;j<properties->getkNN();++j){
            if (labels.row(matches[j].trainIdx).at<int>(0,0) != labels.row(matches[j].queryIdx).at<int>(0,0)){
                l.push_back(labels.row(matches[j].trainIdx));
                vocabulary.push_back(descritores.row(matches[j].trainIdx));
            }
        }
        matches.clear();
    }
    properties->setLabels(l);
    Utils::saveMatrix(properties->getVocPath(), l, "labels-vocabulary");
    return vocabulary;
}

//Encontrar a amostra mais PROXIMA com rotulo DISTINTO do cluster
Mat Dataset::rotuloDistintoMaisProxPorCluster(Mat vocabulary, Mat descritores, Mat l, Mat labels){
    GMatch de;
    vector<GMatch> matches;
    Mat vocAux = vocabulary;
    BOWProperties* properties = BOWProperties::Instance();

    for (int i=0; i<vocAux.rows; i++){
        de.queryIdx = i;
        for (int j=0; j<descritores.rows; j++){
            if (i<j || i!=j){
                de.trainIdx =  j;
                de.distance = Utils::distanciaEuclidiana(descritores.row(i), descritores.row(j));
                matches.push_back(de);
            }
        }
        std::sort(matches.begin(), matches.end(), mysort);
        for(int j=0;matches.size();++j){
            if (labels.row(matches[j].trainIdx).at<int>(0,0) != labels.row(matches[j].queryIdx).at<int>(0,0)){
                l.push_back(labels.row(matches[j].trainIdx));
                vocabulary.push_back(descritores.row(matches[j].trainIdx));
                break;
            }
        }
        matches.clear();
    }
    Utils::saveMatrix(properties->getVocPath(), l, "labels-vocabulary");
    return vocabulary;
}

//Encontrar a amostra mais DISTANTE com rotulo igual do cluster
Mat Dataset::rotuloIgualMaisDistantePorCluster(Mat vocabulary, Mat descritores, Mat l, Mat labels){
    GMatch de;
    vector<GMatch> matches;
    Mat vocAux = vocabulary;
    BOWProperties* properties = BOWProperties::Instance();

    for (int i=0; i<vocAux.rows; i++){
        de.queryIdx = i;
        for (int j=0; j<descritores.rows; j++){
            if (i<j || i!=j){
                de.trainIdx =  j;
                de.distance = Utils::distanciaEuclidiana(descritores.row(i), descritores.row(j));
                matches.push_back(de);
            }
        }
        std::sort(matches.begin(), matches.end(), mysortinv);
        for(int j=0;j<(int)matches.size();++j){
            if (labels.row(matches[j].trainIdx).at<int>(0,0) == labels.row(matches[j].queryIdx).at<int>(0,0)){
                l.push_back(labels.row(matches[j].queryIdx));
                vocabulary.push_back(descritores.row(matches[j].queryIdx));
                break;
            }
        }
        matches.clear();
    }
    properties->setLabels(l);
    Utils::saveMatrix(properties->getVocPath(), l, "labels-vocabulary");
    return vocabulary;
}
