#ifndef examples_h
#define examples_h

#include <stdio.h>
#include <vector>

#include "tiny_dnn/tiny_dnn.h"
#include "pso/psoHeaders.h"
#include "additional_functions.h"
#include "fitnessFunctions.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using std :: cout;
using std :: endl;
using std :: vector;
using std :: string;

int PSOEpochs = 400, PSOBPEpochs = 100, BPEpochs = 500;

/*************** MNIST (no solution) *********************/
void MNISTDigits(int argc, char **argv){
#ifndef NO_MPI
    cout << "MNIST" << endl;
    // Input
    string data_path = argv[1];
    vector<label_t> test_labels;
    vector<label_t> train_labels;
    vector<vec_t> test_images;
    vector<vec_t> train_images;
    parse_mnist_labels(data_path + "/t10k-labels.idx1-ubyte", &test_labels);
    parse_mnist_images(data_path + "/t10k-images.idx3-ubyte", &test_images, 0, 1.0, 2, 2);
    parse_mnist_labels(data_path + "/train-labels.idx1-ubyte", &train_labels);
    parse_mnist_images(data_path + "/train-images.idx3-ubyte", &train_images, 0, 1.0, 2, 2);
    
    train_labels.erase(train_labels.begin() + 2000, train_labels.end());
    train_images.erase(train_images.begin() + 2000, train_images.end());
    test_labels.erase(test_labels.begin() + 857, test_labels.end());
    test_images.erase(test_images.begin() + 857, test_images.end());
    
    
    vector<vec_t> restr_test_labels;
    vector<vec_t> restr_train_labels;
    for(int i = 0; i < test_labels.size(); i++){
        vec_t tmp = {0,0,0,0,0,0,0,0,0,0};
        tmp[test_labels[i]] = 1;
        restr_test_labels.push_back(tmp);
    }
    for(int i = 0; i < train_labels.size(); i++){
        vec_t tmp = {0,0,0,0,0,0,0,0,0,0};
        tmp[train_labels[i]] = 1;
        restr_train_labels.push_back(tmp);
    }
    mapminmax(train_images, 1, -1);
    
    // PSOBP
    vector<int> structure = {1024, 10, 0};
    ParallelCooperativeSwarmParameters sp;
    sp.seed = rand();
    sp.dimension = -1;
    sp.populationSize = 5;
    sp.swarmsNum = 320;
    sp.localVelocityRatio = 1.49;
    sp.globalVelocityRatio = 1.49;
    sp.inirtiaWeight = 2;
    sp.maxInirtiaWeight = 0.9;
    sp.minInirtiaWeight = 0.4;
    sp.maxPSOEpochs = PSOEpochs;
    sp.u_bord = 1;
    sp.l_bord = -1;
    sp.argc = argc;
    sp.argv = argv;
    
    pso_bp<sigmoid, MyFitnessFunctionMSE<sigmoid>, ParallelCooperativeSwarm<MyFitnessFunctionMSE<sigmoid>>, ParallelCooperativeSwarmParameters> _pso_bp(structure);
    _pso_bp.train(train_images, restr_train_labels, test_images, restr_test_labels, sp, PSOBPEpochs);
    network<sequential> nn = _pso_bp.get_nn();
    
    cout << "BackPropagation: " << endl;
    gradient_descent opt;
    opt.alpha *= static_cast<tiny_dnn::float_t>(std::sqrt(train_labels.size()));
    int epoch = 0;
    timer t;
    network<sequential> nn2;
    nn2 << fc<sigmoid>(1024,10,0);
    nn2.fit<mse>(opt, train_images, train_labels, train_labels.size(), BPEpochs,
                 // called for each mini-batch
                 [&](){
                     t.elapsed();
                     t.restart();
                 },
                 // called for each epoch
                 [&](){
                     result res = nn2.test(test_images, test_labels);
                     cout << "Epoch: " << epoch << " " << res.num_success << "/" << res.num_total << " " << 10*nn2.get_loss<mse>(train_images, restr_train_labels) / train_images.size() << endl;
                     epoch++;
                 },1);
    
#endif
}
/*********************************************************/

/************* Diabets Classifier ************************/
void Diabets(int argc, char **argv){
#ifndef NO_MPI
    cout << "Diabets" << endl;
    
    // Input
    string data_path = argv[1];
    vector<vec_t> train_images, test_images;
    vec_t labels;
    readVectorFeautures(data_path, train_images, labels);
    std::vector< vec_t > train_labels, test_labels;
    for(int i = 0; i < labels.size(); i++){
        vec_t tmp(2,0);
        tmp[labels[i]] = 1;
        train_labels.push_back(tmp);
    }
    mapminmax(train_images, 1, -1);
    
    test_images.insert(test_images.begin(), train_images.begin() + 530, train_images.end());
    test_labels.insert(test_labels.begin(), train_labels.begin() + 530, train_labels.end());
    train_images.erase(train_images.begin() + 530, train_images.end());
    train_labels.erase(train_labels.begin() + 530, train_labels.end());
    labels.erase(labels.begin(), labels.end() - 530);
    
    // PSO
    vector<int> structure = {8, 18, 1, 18, 2, 1};
    ParallelCooperativeSwarmParameters sp;
    sp.seed = rand();
    sp.dimension = -1;
    sp.populationSize = 5;
    sp.swarmsNum = 100;
    sp.localVelocityRatio = 1.49;
    sp.globalVelocityRatio = 1.49;
    sp.inirtiaWeight = 2;
    sp.maxInirtiaWeight = 0.9;
    sp.minInirtiaWeight = 0.4;
    sp.maxPSOEpochs = PSOEpochs;
    sp.u_bord = 1;
    sp.l_bord = -1;
    sp.argc = argc;
    sp.argv = argv;
    
    pso_bp<sigmoid, MyFitnessFunctionMSE<sigmoid>, ParallelCooperativeSwarm<MyFitnessFunctionMSE<sigmoid>>, ParallelCooperativeSwarmParameters> _pso_bp(structure);
    _pso_bp.train(train_images, train_labels, test_images, test_labels, sp, PSOBPEpochs);
    network<sequential> nn = _pso_bp.get_nn();
    
    // BP
    vector<label_t> test_label;
    for(int i = 0; i < labels.size(); i++)
        test_label.push_back(labels[i]);
    
    cout << "BackPropagation: " << endl;
    gradient_descent opt;
    opt.alpha *= static_cast<tiny_dnn::float_t>(std::sqrt(train_labels.size()));
    int epoch = 0;
    timer t;
    network<sequential> nn2;
    nn2 << fc<sigmoid>(8,18,1) << fc<sigmoid>(18,2,1);
    nn2.fit<mse>(opt, train_images, train_labels, train_labels.size(), BPEpochs,
                 // called for each mini-batch
                 [&](){
                     t.elapsed();
                     t.restart();
                 },
                 // called for each epoch
                 [&](){
                     result res = nn2.test(test_images, test_label);
                     cout << "Epoch: " << epoch << " " << res.num_success << "/" << res.num_total << " " << 2*nn2.get_loss<mse>(train_images, train_labels) / train_images.size() << endl;
                     epoch++;
                 }, 1);
#endif
}
/*********************************************************/

/**************** Cancer Classifier **********************/
void Cancer(int argc, char **argv){
#ifndef NO_MPI
    cout << "Cancer" << endl;
    // Input
    string data_path = argv[1];
    vector<vec_t> train_images, test_images;
    vec_t labels;
    readVectorFeautures(data_path, train_images, labels);
    std::vector< vec_t > train_labels, test_labels;
    for(int i = 0; i < labels.size(); i++){
        vec_t tmp(2,0);
        tmp[labels[i]] = 1;
        train_labels.push_back(tmp);
    }
    mapminmax(train_images, 1, -1);
    
    test_images.insert(test_images.begin(), train_images.begin() + 489, train_images.end());
    test_labels.insert(test_labels.begin(), train_labels.begin() + 489, train_labels.end());
    train_images.erase(train_images.begin() + 489, train_images.end());
    train_labels.erase(train_labels.begin() + 489, train_labels.end());
    labels.erase(labels.begin(), labels.end() - 489);
    
    // PSO
    vector<int> structure = {9, 15, 1, 15, 2, 1};
    ParallelCooperativeSwarmParameters sp;
    sp.seed = rand();
    sp.dimension = -1;
    sp.populationSize = 5;
    sp.swarmsNum = 91;
    sp.localVelocityRatio = 1.49;
    sp.globalVelocityRatio = 1.49;
    sp.inirtiaWeight = 2;
    sp.maxInirtiaWeight = 0.9;
    sp.minInirtiaWeight = 0.4;
    sp.maxPSOEpochs = PSOEpochs;
    sp.u_bord = 1;
    sp.l_bord = -1;
    sp.argc = argc;
    sp.argv = argv;
    
    pso_bp<sigmoid, MyFitnessFunctionMSE<sigmoid>, ParallelCooperativeSwarm<MyFitnessFunctionMSE<sigmoid>>, ParallelCooperativeSwarmParameters> _pso_bp(structure);
    _pso_bp.train(train_images, train_labels, test_images, test_labels, sp, PSOBPEpochs);
    network<sequential> nn = _pso_bp.get_nn();
    
    // BP
    vector<label_t> test_label;
    for(int i = 0; i < labels.size(); i++)
        test_label.push_back(labels[i]);
    
    cout << "BackPropagation: " << endl;
    gradient_descent opt;
    opt.alpha *= static_cast<tiny_dnn::float_t>(std::sqrt(train_labels.size()));
    int epoch = 0;
    timer t;
    network<sequential> nn2;
    nn2 << fc<sigmoid>(9,15,1) << fc<sigmoid>(15,2,1);
    nn2.fit<mse>(opt, train_images, train_labels, train_labels.size(), BPEpochs,
                 // called for each mini-batch
                 [&](){
                     t.elapsed();
                     t.restart();
                 },
                 // called for each epoch
                 [&](){
                     result res = nn2.test(test_images, test_label);
                     cout << "Epoch: " << epoch << " " << res.num_success << "/" << res.num_total << " " << 2*nn2.get_loss<mse>(train_images, train_labels) / train_images.size() << endl;
                     epoch++;
                 }, 1);
#endif
}
/*********************************************************/

/************** Iris Classifier **************************/
void Iris(int argc, char **argv){
#ifndef NO_MPI
    cout << "Iris" << endl;
    // Input
    string data_path = argv[1];
    vector<vec_t> train_images, test_images;
    vec_t labels;
    readVectorFeautures(data_path, train_images, labels);
    std::vector< vec_t > train_labels, test_labels;
    for(int i = 0; i < labels.size(); i++)
    {
        vec_t tmp(3,0);
        tmp[labels[i]] = 1;
        train_labels.push_back(tmp);
    }
    mapminmax(train_images, 1, -1);
    
    test_images.insert(test_images.begin(), train_images.begin() + 105, train_images.end());
    test_labels.insert(test_labels.begin(), train_labels.begin() + 105, train_labels.end());
    train_images.erase(train_images.begin() + 105, train_images.end());
    train_labels.erase(train_labels.begin() + 105, train_labels.end());
    labels.erase(labels.begin(), labels.end() - 105);
    
    // PSO
    vector<int> structure = {4, 15, 1, 15, 3, 1};
    ParallelCooperativeSwarmParameters sp;
    sp.seed = rand();
    sp.dimension = -1;
    sp.populationSize = 10;
    sp.swarmsNum = 41;
    sp.localVelocityRatio = 1.49;
    sp.globalVelocityRatio = 1.49;
    sp.inirtiaWeight = 2;
    sp.maxInirtiaWeight = 0.9;
    sp.minInirtiaWeight = 0.4;
    sp.maxPSOEpochs = PSOEpochs;
    sp.u_bord = 1;
    sp.l_bord = -1;
    sp.argc = argc;
    sp.argv = argv;
    
    pso_bp<sigmoid, MyFitnessFunctionMSE<sigmoid>, ParallelCooperativeSwarm<MyFitnessFunctionMSE<sigmoid>>, ParallelCooperativeSwarmParameters> _pso_bp(structure);
    _pso_bp.train(train_images, train_labels, test_images, test_labels, sp, PSOBPEpochs);
    network<sequential> nn = _pso_bp.get_nn();
    
    // BP
    vector<label_t> test_label;
    for(int i = 0; i < labels.size(); i++)
        test_label.push_back(labels[i]);
    
    result res = nn.test(test_images, test_label);
    cout << res.num_success << "/" << res.num_total << " " << endl;
    cout << "BackPropagation: " << endl;
    gradient_descent opt;
    opt.alpha *= static_cast<tiny_dnn::float_t>(std::sqrt(train_labels.size()));
    int epoch = 0;
    timer t;
    network<sequential> nn2;
    nn2 << fc<sigmoid>(4,15,1) << fc<sigmoid>(15,3,1);
    nn2.fit<mse>(opt, train_images, train_labels, train_images.size(), BPEpochs,
                 // called for each mini-batch
                 [&](){
                     t.elapsed();
                     t.restart();
                 },
                 // called for each epoch
                 [&](){
                     result res = nn2.test(test_images, test_label);
                     cout << "Epoch: " << epoch << " " << res.num_success << "/" << res.num_total << " " << 3*nn2.get_loss<mse>(train_images, train_labels) / train_images.size() << endl;
                     epoch++;
                 }, 1);
#endif
}
/*********************************************************/

/************** Letters Classifier ***********************/
void Letters(int argc, char **argv){
#ifndef NO_MPI
    cout << "Letters";
    // Input
    string data_path = argv[1];
    vector<vec_t> train_images, test_images;
    vec_t labels;
    readVectorFeautures(data_path, train_images, labels);
    
    train_images.erase(train_images.begin() + 2000, train_images.end());
    labels.erase(labels.begin() + 2000, labels.end());
    
    std::vector< vec_t > train_labels, test_labels;
    for(int i = 0; i < labels.size(); i++){
        vec_t tmp(26,0);
        tmp[labels[i]] = 1;
        train_labels.push_back(tmp);
    }
    vector<label_t> train_labels2;
    for(int i = 0; i < labels.size(); i++)
        train_labels2.push_back(labels[i]);
    
    mapminmax(train_images, 1, -1);
    //mapminmax(train_labels, 1, -1);
    
    test_images.insert(test_images.begin(), train_images.begin() + 1400, train_images.end());
    test_labels.insert(test_labels.begin(), train_labels.begin() + 1400, train_labels.end());
    train_images.erase(train_images.begin() + 1400, train_images.end());
    train_labels.erase(train_labels.begin() + 1400, train_labels.end());
    labels.erase(labels.begin(), labels.end() - 1400);
    
    // PSO
    vector<int> structure = {16, 10, 1, 10, 26, 1};
    ParallelCooperativeSwarmParameters sp;
    sp.seed = rand();
    sp.dimension = -1;
    sp.populationSize = 5;
    sp.swarmsNum = 228;
    sp.localVelocityRatio = 1.49;
    sp.globalVelocityRatio = 1.49;
    sp.inirtiaWeight = 2;
    sp.maxInirtiaWeight = 0.9;
    sp.minInirtiaWeight = 0.4;
    sp.maxPSOEpochs = PSOEpochs;
    sp.u_bord = 1;
    sp.l_bord = -1;
    sp.argc = argc;
    sp.argv = argv;
    
    pso_bp<sigmoid, MyFitnessFunctionMSE<sigmoid>, ParallelCooperativeSwarm<MyFitnessFunctionMSE<sigmoid>>, ParallelCooperativeSwarmParameters> _pso_bp(structure);
    _pso_bp.train(train_images, train_labels, test_images, test_labels, sp, PSOBPEpochs);
    network<sequential> nn = _pso_bp.get_nn();
    
    /*
    // BP
    vector<label_t> test_label;
    for(int i = 0; i < labels.size(); i++)
        test_label.push_back(labels[i]);
    
    cout << "BackPropagation: " << endl;
    gradient_descent opt;
    opt.alpha *= static_cast<tiny_dnn::float_t>(std::sqrt(train_labels.size()));
    int epoch = 0;
    timer t;
    network<sequential> nn2;
    nn2 << fc<sigmoid>(16,10,1) << fc<sigmoid>(10,26,1);
    nn2.fit<mse>(opt, train_images, train_labels, train_labels.size(), BPEpochs,
                 // called for each mini-batch
                 [&](){
                     t.elapsed();
                     t.restart();
                 },
                 // called for each epoch
                 [&](){
                     result res = nn2.test(test_images, test_label);
                     cout << "Epoch: " << epoch << " " << res.num_success << "/" << res.num_total << " " << 26*nn2.get_loss<mse>(train_images, train_labels) / train_images.size() << endl;
                     epoch++;
                 }, 1);*/
#endif
}
/*********************************************************/

void LettersSingle(int argc, char **argv){
    cout << "Letters";
    // Input
    string data_path = argv[1];
    vector<vec_t> train_images, test_images;
    vec_t labels;
    readVectorFeautures(data_path, train_images, labels);
    
    train_images.erase(train_images.begin() + 2000, train_images.end());
    labels.erase(labels.begin() + 2000, labels.end());
    
    std::vector< vec_t > train_labels, test_labels;
    for(int i = 0; i < labels.size(); i++){
        vec_t tmp(26,0);
        tmp[labels[i]] = 1;
        train_labels.push_back(tmp);
    }
    vector<label_t> train_labels2;
    for(int i = 0; i < labels.size(); i++)
        train_labels2.push_back(labels[i]);
    
    mapminmax(train_images, 1, -1);
    //mapminmax(train_labels, 1, -1);
    
    test_images.insert(test_images.begin(), train_images.begin() + 1400, train_images.end());
    test_labels.insert(test_labels.begin(), train_labels.begin() + 1400, train_labels.end());
    train_images.erase(train_images.begin() + 1400, train_images.end());
    train_labels.erase(train_labels.begin() + 1400, train_labels.end());
    labels.erase(labels.begin(), labels.end() - 1400);
    
    // PSO
    vector<int> structure = {16, 10, 1, 10, 26, 1};
    CooperativeSwarmParameters sp;
    sp.seed = rand();
    sp.dimension = -1;
    sp.populationSize = 5;
    sp.swarmsNum = 228;
    sp.localVelocityRatio = 1.49;
    sp.globalVelocityRatio = 1.49;
    sp.inirtiaWeight = 2;
    sp.maxInirtiaWeight = 0.9;
    sp.minInirtiaWeight = 0.4;
    sp.maxPSOEpochs = PSOEpochs;
    sp.u_bord = 1;
    sp.l_bord = -1;
    
    pso_bp<sigmoid, MyFitnessFunctionMSE<sigmoid>, CooperativeSwarm<MyFitnessFunctionMSE<sigmoid>>, CooperativeSwarmParameters> _pso_bp(structure);
    _pso_bp.train(train_images, train_labels, test_images, test_labels, sp, PSOBPEpochs);
    network<sequential> nn = _pso_bp.get_nn();
}



/************* Heart Classifier **************************/
/*void Heart(int argc, char **argv){
#ifndef NO_MPI
    // Input
    string data_path = argv[1];
    vector<vec_t> train_images;
    vec_t labels;
    readVectorFeautures(data_path, train_images, labels);
    std::vector< vec_t > train_labels;
    for(int i = 0; i < labels.size(); i++){
        vec_t tmp(2,0);
        tmp[labels[i]] = 1;
        train_labels.push_back(tmp);
    }
    mapminmax(train_images, 1, -1);
    
    // PSO
    vector<int> structure = {35, 6, 1, 6, 2, 1};
    ParallelCooperativeSwarmParameters sp;
    sp.seed = rand();
    sp.dimension = -1;
    sp.populationSize = 5;
    sp.swarmsNum = 157;
    sp.localVelocityRatio = 1.49;
    sp.globalVelocityRatio = 1.49;
    sp.inirtiaWeight = 2;
    sp.maxInirtiaWeight = 0.9;
    sp.minInirtiaWeight = 0.4;
    sp.maxPSOEpochs = PSOEpochs;
    sp.u_bord = 1;
    sp.l_bord = -1;
    sp.argc = argc;
    sp.argv = argv;
    
    pso_bp<sigmoid, MyFitnessFunctionMSE<sigmoid>, ParallelCooperativeSwarm<MyFitnessFunctionMSE<sigmoid>>, ParallelCooperativeSwarmParameters> _pso_bp(structure);
    _pso_bp.train(train_images, train_labels, sp, PSOBPEpochs);
    network<sequential> nn = _pso_bp.get_nn();
    
    // BP
    vector<label_t> train_label;
    for(int i = 0; i < labels.size(); i++)
        train_label.push_back(labels[i]);
    
    cout << "BackPropagation: " << endl;
    gradient_descent opt;
    opt.alpha *= static_cast<tiny_dnn::float_t>(std::sqrt(train_labels.size()));
    int epoch = 0;
    timer t;
    network<sequential> nn2;
    nn2 << fc<sigmoid>(36,6,1) << fc<sigmoid>(6,2,1);
    nn2.fit<mse>(opt, train_images, train_labels, train_labels.size(), BPEpochs,
                 // called for each mini-batch
                 [&](){
                     t.elapsed();
                     t.restart();
                 },
                 // called for each epoch
                 [&](){
                     result res = nn2.test(train_images, train_label);
                     cout << "Epoch: " << epoch << " " << res.num_success << "/" << res.num_total << " " << 3*nn2.get_loss<mse>(train_images, train_labels) / train_images.size() << endl;
                     epoch++;
                 });
#endif
}*/
/*********************************************************/

/************** Функция Растригина ***********************/
/*void RS(int argc, char **argv){
    CooperativeSwarmParameters sp;
    sp.seed = rand();
    sp.dimension = 10;
    sp.populationSize = 10;
    sp.swarmsNum = 5;
    sp.localVelocityRatio = 1.49;
    sp.globalVelocityRatio = 1.49;
    sp.inirtiaWeight = 2;
    sp.maxInirtiaWeight = 0.9;
    sp.minInirtiaWeight = 0.4;
    sp.maxPSOEpochs = 20000;
    sp.u_bord = 5;
    sp.l_bord = -5;
    
    Rastrigin rf;
    CooperativeSwarm<Rastrigin> test_swarm(rf, sp);
    test_swarm.optimize();
    vec_t v;
    cout << test_swarm.getSolution(v) << endl;
    return;
}*/
/*********************************************************/

/************* Thyroid Classifier ************************/
/*void Thyroid(int argc, char **argv){
#ifndef NO_MPI
    // Input
    string data_path = argv[1];
    vector<vec_t> train_images;
    vec_t labels;
    readVectorFeautures(data_path, train_images, labels);
    std::vector< vec_t > train_labels;
    for(int i = 0; i < labels.size(); i++){
        vec_t tmp(3,0);
        tmp[labels[i]] = 1;
        train_labels.push_back(tmp);
    }
    mapminmax(train_images, 1, -1);
    //mapminmax(train_labels, 1, -1);
    
    // PCPSO-BP
    vector<int> structure = {21, 6, 1, 6, 3, 1};
    ParallelCooperativeSwarmParameters sp;
    sp.seed = rand();
    sp.dimension = -1;
    sp.populationSize = 5;
    sp.swarmsNum = 51;
    sp.localVelocityRatio = 1.49;
    sp.globalVelocityRatio = 1.49;
    sp.inirtiaWeight = 2;
    sp.maxInirtiaWeight = 0.9;
    sp.minInirtiaWeight = 0.4;
    sp.maxPSOEpochs = 10;
    sp.u_bord = 1;
    sp.l_bord = -1;
    sp.argc = argc;
    sp.argv = argv;
    
    pso_bp<sigmoid, MyFitnessFunctionMSE<sigmoid>, ParallelCooperativeSwarm<MyFitnessFunctionMSE<sigmoid>>, ParallelCooperativeSwarmParameters> _pso_bp(structure);
    _pso_bp.train(train_images, train_labels, sp, 200);
    network<sequential> nn = _pso_bp.get_nn();
    
    // BP
    vector<label_t> train_label;
    for(int i = 0; i < labels.size(); i++)
        train_label.push_back(labels[i]);
    
    cout << "BackPropagation: " << endl;
    gradient_descent opt;
    opt.alpha *= static_cast<tiny_dnn::float_t>(std::sqrt(train_labels.size()));
    int epoch = 0;
    timer t;
    network<sequential> nn2;
    nn2 << fc<sigmoid>(21,6,1) << fc<sigmoid>(6,3,1);
    nn2.fit<mse>(opt, train_images, train_labels, train_labels.size(), 200,
                 // called for each mini-batch
                 [&](){
                     t.elapsed();
                     t.restart();
                 },
                 // called for each epoch
                 [&](){
                     result res = nn2.test(train_images, train_label);
                     cout << "Epoch: " << epoch << " " << res.num_success << "/" << res.num_total << " " << 3*nn2.get_loss<mse>(train_images, train_labels) / train_images.size() << endl;
                     epoch++;
                 });
#endif
}*/
/*********************************************************/

#endif /* examples_h */
