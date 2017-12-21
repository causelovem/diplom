#ifndef additional_functions
#define additional_functions

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <dirent.h>

#include "pso/psoHeaders.h"
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using std :: cout;
using std :: endl;

int getdir (std::string dir, std::vector<std::string> &files){
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }
    
    while ((dirp = readdir(dp)) != NULL) {
        if(std::string(dirp->d_name) == ".")
            continue;
        if(std::string(dirp->d_name) == "..")
            continue;
        if(std::string(dirp->d_name) == ".DS_Store")
            continue;
        
        files.push_back(dir + "/" + std::string(dirp->d_name));
    }
    closedir(dp);
    
    return 0;
}

int readFiles(std::string dir, std::vector< std::vector<std::string> > &out){
    std::vector<std::string> folders;
    if(getdir(dir, folders))
        return 1;
    
    out.resize(folders.size());
    
    for(int i = 0; i < folders.size(); i++)
    {
        if(getdir(folders[i], out[i]))
            return 1;
    }
    
    return 0;
}

/* Number of inputs, number of outputs, bias_flag, ... */
void buildNetParams(const network<sequential> &nn, std::vector<int> &structure){
    for(int i = 0; i < nn.depth(); i++){
        structure.push_back(nn[i]->in_size());
        structure.push_back(nn[i]->out_size());
        if(nn[i]->weights().size() == 1)
            structure.push_back(0);
        else
            structure.push_back(1);
    }
}

void encodeNeuralNetwork(network<sequential> &nn, vec_t &codeOfNN){
    
    for (int i = 0; i < nn.depth(); i++) {
        std::vector<vec_t*> weights = nn[i]->weights();
        codeOfNN.insert(codeOfNN.end(), weights[0]->begin(), weights[0]->end());
        if(weights.size() != 1)
            codeOfNN.insert(codeOfNN.end(), weights[0]->begin(), weights[0]->end());
    }
}

void decodeNeuralNetwork(const vec_t &codeOfNN, network<sequential> _nn){
    int tmp = 0;
    for (int i = 0; i < _nn.depth(); i++)
    {
        std::vector<tiny_dnn::tensor_t> data;
        tiny_dnn::tensor_t tens;
        tens.resize(_nn[i]->in_size());
        for(int j = 0; j < _nn[i]->in_size(); j++)
        {
            tens[j].resize(_nn[i]->out_size());
            for(int k = 0; k < _nn[i]->out_size(); k++)
                tens[j][k] = codeOfNN[tmp++];
        }
        data.push_back(tens);
        
        int m = 0;
        for(int j = 0; j < _nn[i]->in_size(); j++)
        {
            for(int k = 0; k < _nn[i]->out_size(); k++)
            {
                (*(_nn[i]->weights()[0]))[m++] = data[0][j][k];
            }
        }
        
        if(_nn[i]->weights().size() == 2)
        {
            tens.clear();
            tens.resize(1);
            tens[0].resize(_nn[i]->out_size());
            for(int k = 0; k < _nn[i]->out_size(); k++)
                tens[0][k] = codeOfNN[tmp++];
            data.push_back(tens);
            
            m = 0;
            for(int k = 0; k < _nn[i]->out_size(); k++)
            {
                (*(_nn[i]->weights()[1]))[m++] = data[1][0][k];
            }
        }
    }
}

template <class Activate>
vec_t get_result(const vec_t &codeOfNN,const vec_t &input, std::vector<int> *_structure, Activate _activate_function){
    int start = 0;
    tiny_dnn::vec_t new_input = input;
    vec_t res;
    
    for(int k = 0; k < _structure->size(); k+=3)
    {
        int bias_flag = (*_structure)[k+2];
        res.resize((*_structure)[k+1], 0);
        
        for(int i = 0; i < (*_structure)[k+1]; i++)
        {
            res[i] = 0.0;
            for (int j = 0; j < (*_structure)[k]; j++)
                res[i] += new_input[j] * codeOfNN[start + j * (*_structure)[k+1] + i];
            
            if(bias_flag)
                res[i] += codeOfNN[start + (*_structure)[k] * (*_structure)[k+1] + i];
            
            res[i] = _activate_function.f(res, i);
        }
        start += ((*_structure)[k] + bias_flag) * (*_structure)[k+1];
        new_input = res;
    }
    
    return res;
}

int readVectorFeautures(std::string datapath, std::vector<vec_t> &data, vec_t &labels){
    std::ifstream f(datapath);
    int count, sizeVec;
    f >> count;
    f >> sizeVec;
    data.resize(count);
    labels.resize(count);
    
    for(int i = 0; i < count; i++)
    {
        f >> labels[i];
        data[i].resize(sizeVec);
        for(int j = 0; j < sizeVec; j++)
            f >> data[i][j];
    }
    
    return 0;
}

void mapminmax(std::vector<vec_t> &v, double ymax, double ymin){
    size_t m = v[0].size();
    for(int i = 0; i < m; i++)
    {
        double xmax = v[0][i], xmin = v[0][i];
        for(int j = 0; j < v.size(); j++)
        {
            if(xmax < v[j][i])
                xmax = v[j][i];
            
            if(xmin > v[j][i])
                xmin = v[j][i];
        }
        
        if(xmax == xmin)
            for(int j = 0; j < v.size(); j++)
                v[j][i] = 0;
        else
            for(int j = 0; j < v.size(); j++)
                v[j][i] = (ymax-ymin)*(v[j][i]-xmin)/(xmax-xmin) + ymin;
    }
}

#endif
