#include "Utils.h"
#include "Dataset.h"
#include <QCoreApplication>
#include <iostream>
#include <sys/time.h>
#include "BOWProperties.h"
#include <opencv2/nonfree/nonfree.hpp>
#include <iosfwd>
#include <QVector>
#include <fstream>
#include <sys/stat.h>
#include <string.h>
#include <string>
#include <cstdlib>
#include <sstream>
#include <sys/types.h>
#include <opencv2/core/core.hpp>
#include <sstream>


std::ostringstream s;
ofstream out;

void help(){
    cout << "Entrada de parametros errada." << endl;
    cout << "Os parametros de entrada devem seguir o seguinte formato:" << endl;
    cout << " - Caminho da pasta do banco de imagens" << endl;
    cout << " - Caminho da pasta dos arquivos de saida (Os arquivos já computados no processo que vão ser usados novamente devem estar dentro dessa pasta)" << endl;
    cout << " - Tipo do detector de pontos de interesse (SIFT,SURF, DENSE, RANDOM )" << endl;
    cout << " - Tipo do descritor dos pontos de interesse (SIFT,SURF, LBP, BIC, HOG, PHOG, BOC)" << endl;
    cout << " - Tipo de criação do vocabulário de palavras (1 - k-means, 2-U-OPF, 3- k-means com bordas (centroide+fronteira), 4 - k-means com bordas (fronteira), 5 U-OPF com bordas (centroide+fronteira), 6 U-OPF com bordas (fronteira)" << endl;
    cout << " - Número de clusters a serem criados (não necessário para U-OPF)" << endl;
    cout << " - Número de vizinhos a serem observados para a borda (necessário somente para metodologia com bordas)" << endl;
    cout << " Aperte qualquer tecla para continuar..." << endl;
    cin.get();
}

void run(string folder, string output, int type_voc_building, int k, int clusterCount, string type_detector,
         string type_descritor, string input, string outputResults, int ncluster, string outputRoot, string vocPath, float t){
    cout << "Initializing" << endl;
    initModule_nonfree();
    Dataset dataset = Dataset(folder);

    BOWProperties* properties = BOWProperties::Instance();
    properties->setFeatureDetector(type_detector);
    properties->setBOWTrainer(clusterCount);
    properties->setDescriptorMatcher("BruteForce");
    properties->setPathOutput(output);
    properties->setDescriptorExtractor(type_descritor);
    properties->setTypeVocabularyBuilding(type_voc_building);
    properties->setkNN(k);
    properties->setPathInput(input);
    properties->setPathOutResults(outputResults);
    properties->setnCluster(ncluster);
    properties->setOutputRoot(outputRoot);
    properties->setVocPath(vocPath);
    properties->setThreshold(t);

    dataset.trainBOW();
    cout << "Vocabulary computed" << endl;
    dataset.trainClassifier();
    cout << "Classifiers trained" << endl;
    cout << "OPF Classifiers" << endl;
    dataset.classifier();
}

int main(int argc, char *argv[]){
    QCoreApplication a(argc, argv);

    if (argc <5 || argc > 7){
        help();
        return 1;
    }

    string folder;
    string output;
    string outputResults;
    string outputRoot;
    string vocPath;
    string input;
    string type_detector;
    string type_descriptor;
    int type_voc_building;
    int clusterCount=0;
    int clusterMoment=0;
    float t;
    int knn=0;

    folder=argv[1];
    output=argv[2];
    type_detector = argv[3];
    type_descriptor = argv[4];
    type_voc_building = atoi(argv[5]);
    clusterMoment = atoi(argv[6]);
    clusterCount = atoi(argv[7]);
    t = atof (argv[8]);

    cout << endl;
    cout << "folder: " << folder << endl;
    cout << "output file: " << output << endl;
    cout << "type detector: " << type_detector << endl;
    cout << "type descriptor: " << type_descriptor << endl;
    cout << "type building vocabulary: " << type_voc_building << endl;
    s << clusterCount;
    cout << "cluster count: " << s.str() << endl;
    s.str("");
    s << clusterMoment;
    cout << "Cluster moment: " << s.str() << endl;
    s.str("");   s << t;
    cout << "Threshold: " << s.str() << endl;

    s.str("");
    string pasta = output+type_detector+"-"+type_descriptor+"/";
    mkdir(pasta.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    s << t;
    pasta.append(s.str()).append("/");
    outputRoot = pasta;
    mkdir(pasta.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    s.str("");
    switch (type_voc_building) {
    case 0: pasta.append("Random/"); break;
    case 2: pasta.append("U-OPF/"); break;
    default: pasta.append("k-means/"); break;
    }
    mkdir(pasta.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    input = pasta;

    s << type_voc_building;
    pasta.append(s.str()).append("/");
    mkdir(pasta.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    s.str(""); s << clusterCount;
    pasta.append(s.str()).append("/");
    outputResults = pasta;
    input.append("dados-").append(s.str()).append("-");
    mkdir(pasta.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);


    s.str("");
    s<< clusterMoment;
    pasta.append(s.str()).append("/");
    mkdir(pasta.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    input.append(s.str()).append("/");
    mkdir(input.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    vocPath = pasta;
    output = pasta;

    run(folder, output, type_voc_building, knn, clusterCount, type_detector, type_descriptor, input, outputResults, clusterMoment, outputRoot, vocPath, t);
    return 0;
}
