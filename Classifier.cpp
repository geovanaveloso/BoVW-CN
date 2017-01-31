#include "Classifier.h"
#include <BOWProperties.h>
#include <string>
#include <fstream>
#include <sys/stat.h>
#include <string.h>
#include <cstdlib>
#include "Utils.h"
#include <locale.h>
extern "C" {
#include "OPF.h"
}

using namespace std;
std::ostringstream sr;

Classifier::~Classifier()
{
}

void Classifier::classifier(){
    BOWProperties* properties = BOWProperties::Instance();
    string name = properties->getPathOutput()+"histograms-original";
    txt2opf(name);

    Subgraph *g = NULL;
    string path = name+".dat";

    g = ReadSubgraph(&path[0]);

    float acc=0;
    string outpasta = properties->getPathOutput()+"/";
    Subgraph *treinamento = NULL, *teste = NULL;
    opf_SplitSubgraph(g, &treinamento, &teste,  0.8);
    string write1 = outpasta+"treinamento.dat";
    WriteSubgraph(treinamento,&write1[0]);
    string write2 = outpasta+"teste.dat";
    WriteSubgraph(teste,&write2[0]);

    opf_OPFTraining(treinamento);
    opf_OPFClassifying(treinamento, teste);
    acc = opf_Accuracy(teste);

    ofstream outResults;
    sr.str("");               sr << properties->getTypeVocabularyBuilding();
    string results = properties->getPathOutResults()+properties->getType_detector()+"-"+
            properties->getType_descritor()+"-"+sr.str()+"-";
    if (properties->getTypeVocabularyBuilding()>=3 ){
        sr.str("");               sr << properties->getkNN();
        results+=sr.str();
    }
    results+=".txt";
    outResults.open(results.c_str(), ios::app);
    outResults <<  properties->getkCluster() << ";" << properties->getnCluster() << ";" << (acc*100) << endl;
    DestroySubgraph(&teste);
    DestroySubgraph(&treinamento);
    DestroySubgraph(&g);
}

void  Classifier::txt2opf(string name){
    std::setlocale(LC_ALL, "en_US.UTF-8");
    FILE *fpIn = NULL,*fpOut = NULL;
    int n, ndata, nclasses, i,j, id,label;
    float aux = 0.0;
    string namein = name;
    string nameout = name+".dat";

    fpIn = fopen(namein.c_str(),"r");
    fpOut = fopen(nameout.c_str(),"wb");

    /*writting the number of samples*/
    if (fscanf(fpIn,"%d",&n) != 1) {
      fprintf(stderr,"Could not read number of samples");
      exit(-1);
    }
    fwrite(&n,sizeof(int),1,fpOut);

    /*writting the number of classes*/
    if (fscanf(fpIn,"%d",&nclasses) != 1) {
      fprintf(stderr,"Could not read number of classes");
      exit(-1);
    }

    fwrite(&nclasses,sizeof(int),1,fpOut);

    /*writting the number of features*/
    if (fscanf(fpIn,"%d",&ndata) != 1) {
      fprintf(stderr,"Could not read number of features");
      exit(-1);
    }

    fwrite(&ndata,sizeof(int),1,fpOut);

    /*writting data*/

    for(i = 0; i < n; i++)	{
        if (fscanf(fpIn,"%d",&id) != 1) {
            fprintf(stderr,"Could not read sample id");
            exit(-1);
        }
        fwrite(&id,sizeof(int),1,fpOut);

        if (fscanf(fpIn,"%d",&label) != 1) {
            fprintf(stderr,"Could not read sample label - %d", i);
            exit(-1);
        }
        fwrite(&label,sizeof(int),1,fpOut);


        for(j = 0; j < ndata; j++){
           if (fscanf(fpIn," %f", &aux) != 1) {
                fprintf(stderr,"Could not read sample features");
               exit(-1);
            }
            fwrite(&aux,sizeof(float),1,fpOut);
        }
    }

    fclose(fpIn);
    fclose(fpOut);
}

void Classifier::opf2txt(string name){
    FILE *fpIn = NULL;
    int n, ndata, nclasses, label, i,j, id;
    float aux;

    string namein = name+".dat";
    string name_arff = name+".arff";
    ofstream outARFF;
    outARFF.open(name_arff.c_str(), ios::app);

    fpIn = fopen(namein.c_str(),"rb");

     outARFF << "@RELATION " << name << endl;

    /*gravando numero de objetos*/
    fread(&n,sizeof(int),1,fpIn);

    /*gravando numero de classes*/
    fread(&nclasses,sizeof(int),1,fpIn);

    /*gravando tamanho vetor de caracteristicas*/
    fread(&ndata,sizeof(int),1,fpIn);
    for (int p =0; p<ndata; p++){
        outARFF << "@ATTRIBUTE at" << p << " REAL" << endl;
    }
     outARFF << "@ATTRIBUTE class {";
    for (int p =0; p<nclasses; p++){
        if (p==nclasses-1){
            outARFF << p;
        }else{
            outARFF << p << ", ";
        }
    }
    outARFF << "}\n" << "@DATA" << endl;
    for(i = 0; i < n; i++){
        fread(&id,sizeof(int),1,fpIn);
        fread(&label,sizeof(int),1,fpIn);
        for(j = 0; j < ndata; j++){
            fread(&aux,sizeof(float),1,fpIn);
            outARFF << aux << ", ";
        }
         outARFF << label << endl;
    }
    fclose(fpIn);
}

vector<int> Classifier::vocabularyWithOPF(string name){
    Subgraph *g = NULL;
    BOWProperties* properties = BOWProperties::Instance();
    string nameFull = properties->getPathOutput()+name;
    txt2opf(nameFull);
    vector<int> roots;
    string path = nameFull+".dat";
    int root;
    bool rep;
    Mat labels;

    g = ReadSubgraph(&path[0]);
    cout << "founding best k"<< endl;
    opf_BestkMinCut(g,1,properties->getkCluster());

    opf_ElimMaxBelowH(g, (int)(0.1*g->nnodes));
    opf_OPFClustering(g);

    int prop=0;
    for (int i = 0; i < g->nnodes; i++){
        root = g->node[i].root;
        rep=true;

            for (int j = 0; j<(int)roots.size(); j++){
                   if (root == roots.at(j)){
                      rep = false;
                   }
            }

            if(root!=0){
                prop++;
            }

        if(rep){
             roots.push_back(root);
        }
        labels.push_back(g->node[i].label);
    }
    Utils::saveMatrix(properties->getPathInput(), labels, "labels");
    DestroySubgraph(&g);
    return roots;
}
