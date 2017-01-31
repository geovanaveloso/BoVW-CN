#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <string>
#include <vector>

using namespace std;

class Classifier
{
public:
    ~Classifier();
    void txt2opf(string name);
    void opf2txt(string name);
    void splitFolds();
    void classifier();
    vector<int> vocabularyWithOPF(string name);
};

#endif // CLASSIFIER_H
