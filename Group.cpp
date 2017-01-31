#include "Group.h"
#include "Utils.h"
#include <omp.h>
#include <QVector>

Group::Group(string path, string name) {
    this->path = path;
    this->name = name;
    QVector<string> imageNames;
    string pathtrain = path+"/train/";
    string pathtest = path+"/test/";
    imageNames = Utils::getFileNames((pathtrain).c_str(), imageNames);
    foreach (string imageName, imageNames){
        images.push_back(Image(pathtrain+imageName, true));
    }
    imageNames.clear();
    imageNames = Utils::getFileNames((pathtest).c_str(), imageNames);
    foreach (string imageName, imageNames){
        images.push_back(Image(pathtest+imageName, false));
    }
}

Group::~Group() {
}

//Get path
string Group::getPath(){
    return this->path;
}

//Get name group
string Group::getName(){
    return this->name;
}

/*
Treinamento BOW do grupo. Isso pega os descritores da imagem e os coloca em trainer BOW.
*/
void Group::trainBOW() {
    Ptr<BOWKMeansTrainer> trainer = BOWProperties::Instance()->getBowTrainer();
    for (int i = 0; i < (int)images.size(); i++){
        if (images[i].isTrain()){
            Mat descriptors = images[i].getDescriptors();
                if (!descriptors.empty()){
                    trainer->add(descriptors);
                }
         }
    }
}

int Group::getHistograms(int n, int g, string filename) {
    for (int i = 0; i < (int)images.size(); i++){
        if (!images[i].isTrain()){
            Mat imageHistogram = images[i].extractHistogram();
                Utils::writeHistograms(imageHistogram, g, n,filename);
                n++;
           }
    }
    return n;
}
