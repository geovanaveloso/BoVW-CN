#include "Graph.h"
#include <opencv2/opencv.hpp>
#include <igraph/igraph.h>

using namespace cv;

Graph::Graph(){
}

void Graph::addVertex(Mat image){
    int x=0;
    for (int i=0; i < image.rows; i++){
        for (int j=0; j<image.cols; j++){
            vertices.push_back(Vertex(x, i, j, image.at<uchar>(i,j)));
            x++;
            if (image.at<uchar>(i,j)>this->L){
                this->L = image.at<uchar>(i,j);
            }
        }
    }    
    if (vertices.size()%2!=0){
        vertices.erase(vertices.end());
    }
    addEdges(3);
}

Graph Graph::thresholds(Graph g, float f){
    Graph graph = g;

    for (int i=0; i<(int) graph.edges.size(); i++){
        Edge e = graph.edges.at(i);
        if (e.weight>f){
            graph.edges.erase(graph.edges.begin()+i);
            i--;
        }
    }
    return graph;
}

void Graph::release(){
    vertices.clear();
    edges.clear();
}

void Graph::addEdges(int r){
    float peso=0;

    for (int i=0; i<(int)vertices.size(); i++){
        for (int j=i+1; j<(int)vertices.size(); j++){
            if (distanciaEuclidiana(vertices.at(i), vertices.at(j))<=r){
                peso = calcularPeso2(vertices.at(i), vertices.at(j), r);
                edges.push_back(Edge(vertices.at(i), vertices.at(j), peso));
            }
        }
    }
}

float Graph::calcularPeso1(Vertex v1, Vertex v2, int r){
    float peso = pow((v1.x-v2.x),2)+pow((v1.y-v2.y),2)+pow((v1.valor_pixel-v2.valor_pixel),2);
    peso/=pow(255,2)+pow(r,2);
    return peso;
  }

float Graph::calcularPeso2(Vertex v1, Vertex v2, int r){
    float peso = pow((v1.x-v2.x),2)+pow((v1.y-v2.y),2)+pow(r,2);
    float peso2 = abs(v1.valor_pixel-v2.valor_pixel);
    peso2/=L;
    peso*=peso2;
    peso/=pow(r,2)+pow(r,2);
    return peso;
  }

float Graph::distanciaEuclidiana(Vertex v1, Vertex v2){
    float result = 0, x1 = v1.x, x2 = v2.x, y1 = v1.y, y2 = v2.y;
    result = pow((x1-x2),2) + pow((y1-y2),2);
    return sqrt(result);
}

Mat Graph::extractFeatures(Graph graph){
    vector<float> resp, aux;
    igraph_t g;
    igraph_vector_t v;

    // add vertices
    igraph_vector_init(&v,(int)graph.vertices.size());

    // add arestas
    for (int j=0; j<(int)graph.edges.size(); j++){
        Edge e = graph.edges.at(j);
        VECTOR(v)[e.startVertex.id]=e.endVertex.id;
    }
    //criar grafo
    igraph_create(&g,&v,0,IGRAPH_UNDIRECTED);

    resp.push_back(extractAvgBetweenness(g));
    resp.push_back(extractNumberCommunity(g));
    resp.push_back(extractAvgDegree(g, v));
    resp.push_back(extractTransitivy(g));
    resp.push_back(extractAvgShortestPath(g));
    aux = extractMotifs(g,3);
    for (int i =0; i<(int)aux.size(); i++){
        resp.push_back(aux.at(i));
    }
    aux  = extractMotifs(g,4);
    for (int i =0; i<(int)aux.size(); i++){
        resp.push_back(aux.at(i));
    }
    aux = extractConnectivityHist(g, v);
    for (int i =0; i<(int)aux.size(); i++){
        resp.push_back(aux.at(i));
    }

    Mat descritores(resp, true);
    descritores = descritores.reshape(1, 1);
    igraph_vector_destroy(&v);
    igraph_destroy(&g);
    resp.clear(); aux.clear();
    return descritores;
}

Mat Graph::extractFeatures1(Graph graph){
    Mat descritores;
    vector<float> pi;
    float contraste=0.0, energia=0.0, entropia=0.0;

    pi = extractHist(graph);

    for (float i=0.0; i<pi.size(); i++){
        contraste+=pi.at(i)* pow(i,2);
        energia+= pow(pi.at(i), 2);
        if (pi.at(i)!=0){
            entropia+=pi.at(i)*log2(pi.at(i));
        }
    }

    entropia = -entropia;

    if (isnan(contraste)!=0){
        contraste=0;
    }
    if (isnan(energia)!=0){
        energia=0;
    }
    if (isnan(entropia)!=0){
        entropia=0;
    }

    descritores.push_back(contraste);
    descritores.push_back(energia);
    descritores.push_back(entropia);
    descritores = descritores.reshape(descritores.channels(),1);
    return descritores;
}

vector<float> Graph::extractHist(Graph g){
    vector<float> hist;
    vector<float> pi;
    float vet[g.vertices.size()];
    float f = 0.0;

    for(int i = 0; i < g.vertices.size(); i++) {
        vet[i] = 0;
    }

    for (int i=0; i<g.edges.size(); i++){
        Edge e = g.edges[i];
        vet[e.startVertex.id]++;
    }

    for (int i = 0; i < g.vertices.size(); ++i) {
        hist.push_back(vet[i]);
    }

    for (int i=0; i<hist.size(); i++){
        f = hist.at(i)/ g.edges.size();
        pi.push_back(f);
    }
    return pi;
}

vector<float> Graph::extractConnectivityHist(igraph_t g, igraph_vector_t v){
    vector<float> hist;
    float vet[20];
    int f;
    for(int i = 0; i < 20; i++) {
        vet[i] = 0;
    }
    igraph_degree(&g, &v, igraph_vss_all(), IGRAPH_ALL, IGRAPH_LOOPS);
    for (long int i=0; i<igraph_vector_size(&v); i++) {
        f = VECTOR(v)[i];
        if (f<=20){
            vet[f]++;
        }
    }
    for (int i = 0; i < 20; ++i) {
        hist.push_back(vet[i]);
    }
    return hist;
}

float Graph::extractAvgShortestPath(igraph_t g){
    igraph_real_t res;
    igraph_average_path_length(&g, &res, IGRAPH_UNDIRECTED, 1);
    return (float) res;
}

float Graph::extractAvgBetweenness(igraph_t g){
    igraph_vector_t res;
    float avg = 0;
    igraph_vector_init(&res, 0);
    igraph_betweenness(&g, &res, igraph_vss_all(), false, 0, false);
    for (long int i=0; i<igraph_vector_size(&res); i++) {
        avg += VECTOR(res)[i];
    }
    avg/=igraph_vector_size(&res);
    igraph_vector_destroy(&res);
    return avg;
}

float Graph::extractNumberCommunity(igraph_t g){
    igraph_matrix_t merges;
    igraph_vector_t modularity;
    igraph_vector_init(&modularity, 0);
    igraph_matrix_init(&merges, 0, 0);
    igraph_community_walktrap(&g, 0, 10, &merges, &modularity, 0);
    float f = igraph_matrix_nrow(&merges);
    igraph_vector_destroy(&modularity);
    igraph_matrix_destroy(&merges);
    return f;
}

float Graph::extractAvgDegree(igraph_t g, igraph_vector_t v){
    float avg = 0;
    igraph_degree(&g, &v, igraph_vss_all(), IGRAPH_ALL, IGRAPH_LOOPS);
    for (long int i=0; i<igraph_vector_size(&v); i++) {
        avg+=(float) VECTOR(v)[i];
    }
    avg/=igraph_vector_size(&v);
    return avg;
}

float Graph::extractTransitivy(igraph_t g){
    igraph_real_t res;
    igraph_transitivity_undirected(&g, &res, IGRAPH_TRANSITIVITY_ZERO);
    return (float) res;
}

vector<float> Graph::extractMotifs(igraph_t g, int n_motis){
    vector<float> d;
    igraph_vector_t res, cp;
    int i, n;
    igraph_real_t sum=0.0;
    igraph_vector_init(&res, 0);
    igraph_vector_init_real(&cp, 8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    igraph_motifs_randesu(&g, &res, n_motis, &cp);
    n=igraph_vector_size(&res);

    for (i=0; i<n; i++) {
        if (!igraph_is_nan(VECTOR(res)[i])) {
            sum += VECTOR(res)[i];
        }
    }
    for (i=0; i<n; i++) {
        float f = VECTOR(res)[i]/sum;
        if (isnan(f)){
            f=0;
        }
        d.push_back(f);
    }
    igraph_vector_destroy(&res);
    igraph_vector_destroy(&cp);
    return d;
}

