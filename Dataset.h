#ifndef DATASET_H
#define DATASET_H

#include "Group.h"
#include "Utils.h"
#include "BOWProperties.h"
#include <QVector>

class Dataset {
public:
    Dataset(string folder);
    ~Dataset();
    QVector<string> classNames;

    void trainBOW();
    void trainClassifier();
    void classifier();
    void writeCabecalhoOPF(string filename, int n_caracteristicas);
    void writeCabecalhoARFF(string filename);
    void criar_matriz_distancia(Mat descritores);
    Mat trainerOPF(Mat descritores);
    Mat bordasPorClusters(Mat vocabulary, Mat descritores, Mat l, Mat labels);
    Mat trainerRandom(Mat descritores);
    Mat trainerKmeans(Mat descritores);
    Mat rotuloDistintoMaisProxPorCluster(Mat vocabulary, Mat descritores, Mat l, Mat labels);
    Mat rotuloIgualMaisDistantePorCluster(Mat vocabulary, Mat descritores, Mat l, Mat labels);

private:
    string path;
    std::vector<Group> groups;


};

#endif
