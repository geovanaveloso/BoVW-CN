#ifndef GRAPH_H
#define GRAPH_H
#include <opencv2/opencv.hpp>
#include <igraph/igraph.h>

using namespace cv;
using namespace std;

struct Vertex {
    Vertex(int id, int x, int y, int valor_pixel): id(id), x(x), y(y),  valor_pixel(valor_pixel) { }
    ~Vertex() { }

    int id;
    int x;
    int y;
    int valor_pixel;
};

struct Edge {
    Edge(Vertex startVertex, Vertex endVertex, float weight):
        startVertex(startVertex), endVertex(endVertex), weight(weight) { }
    ~Edge() { }
    Vertex startVertex;
    Vertex endVertex;
    float weight;
};

class Graph
{
public:
    Graph(Mat image);
    Graph();
    void addVertex(Mat image);
    void addEdges(int r);
    Graph thresholds(Graph g, float f);
    float calcularPeso1(Vertex v1, Vertex v2, int r);
    float calcularPeso2(Vertex v1, Vertex v2, int r);
    float distanciaEuclidiana(Vertex v1, Vertex v2);
    vector<Vertex> vertices;
    vector<Edge> edges;
    igraph_t construirIgraph(Graph graph);
    Mat extractFeatures(Graph graph);
    vector<float> extractMotifs(igraph_t g, int n_motis);
    float extractTransitivy(igraph_t g);
    float extractAvgDegree(igraph_t g, igraph_vector_t v);
    float extractNumberCommunity(igraph_t);
    float extractAvgBetweenness(igraph_t g);
    float extractAvgShortestPath(igraph_t g);
    vector<float> extractConnectivityHist(igraph_t g, igraph_vector_t v);
    void release();
    Mat extractFeatures1(Graph graph);
    vector<float> extractHist(Graph g);
private:
    int L=0;

};

#endif // GRAPH_H
