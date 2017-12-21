#ifndef pso_bp_hpp
#define pso_bp_hpp

#include <iostream>
#include <vector>
#include "../tiny-dnn/tiny_dnn/tiny_dnn.h"

template<class Activate, class FitnessFunction, class PSO, class PSOParameters>
class pso_bp{
    tiny_dnn::network<tiny_dnn::sequential> _nn;
    
    std::vector<int> _structure;
    std::vector<vec_t> _train_data;
    std::vector<vec_t> _train_labels;
    
    FitnessFunction _fitnessFunction;
    
    void decodeNeuralNetwork(const vec_t &codeOfNN){
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
    
public:
    pso_bp(const std::vector<int> &structure):_structure(structure){
        
        /* Number of inputs, number of outputs, bias_flag, ... */
        
        assert((structure.size() % 3 == 0) && "Structure's vector shall be divided by 3" );
        
        for(int i = 0; i < structure.size(); i+=3)
            _nn << tiny_dnn::fully_connected_layer<Activate>(structure[i],structure[i+1],structure[i+2]);
        
        _fitnessFunction._structure = &_structure;
    }
    
    void train(const std::vector< vec_t > &train_data, const std::vector< vec_t > &train_labels, const std::vector< vec_t > &test_data, const std::vector< vec_t > &test_labels, PSOParameters pp, int bpEpochs){
        
        assert(train_data.size() == train_labels.size());
        assert(train_data[0].size() == _structure[0]);
        assert(train_labels[0].size() == _structure[_structure.size() - 2]);
        
        _train_data = train_data;
        _train_labels = train_labels;
        
        _fitnessFunction._train_data = &_train_data;
        _fitnessFunction._train_labels = &_train_labels;
        
        vector<tiny_dnn::label_t> train_labels2;
        for (int i = 0; i < train_labels.size(); i++) {
            int max = 0;
            for(int j = 0; j < train_labels[i].size(); j++){
                if(train_labels[i][j] > train_labels[i][max])
                    max = j;
            }
            
            train_labels2.push_back(max);
        }
        
        vector<tiny_dnn::label_t> test_labels2;
        for (int i = 0; i < test_labels.size(); i++) {
            int max = 0;
            for(int j = 0; j < test_labels[i].size(); j++){
                if(test_labels[i][j] > test_labels[i][max])
                    max = j;
            }
            
            test_labels2.push_back(max);
        }
        
        
        // PSO
        std::cout << "PSO:" << std::endl;
        int dimension = 0;
        for(int i = 0; i < _structure.size(); i+=3)
            dimension += (_structure[i]+_structure[i+2]) * _structure[i+1];
        pp.dimension = dimension;
        
        unsigned long t1 = clock();
        PSO _pso(_fitnessFunction, pp);
        _pso.optimize();
        vec_t solution;
        _pso.getSolution(solution);
        decodeNeuralNetwork(solution);
        unsigned long t2 = clock();
        
        tiny_dnn::result res;
        //res = _nn.test(test_data, test_labels2);
        std::cout << "PSO results: " << res.num_success << "/" << res.num_total << " Time: " << (t2 - t1)/CLOCKS_PER_SEC << std::endl;
        
        // BP
        std::cout << "PSOBP:" << std:: endl;
        t1 = clock();
        tiny_dnn::gradient_descent optimizer; size_t minibatch_size = 1;
        
        int epoch = 0;
        tiny_dnn::timer t;
        _nn.train<tiny_dnn::mse>(optimizer, train_data, train_labels2, minibatch_size, bpEpochs,
                     // called for each mini-batch
                     [&](){
                         t.elapsed();
                         t.restart();
                     },
                     // called for each epoch
                     [&](){
                         //tiny_dnn::result res = _nn.test(test_data, test_labels2);
                         epoch++;
                         std::cout << "Epoch: " << epoch << " " << res.num_success << "/" << res.num_total << " " << _structure[_structure.size()-2]*_nn.get_loss<tiny_dnn::mse>(train_data, train_labels) / train_data.size() << std::endl;
                     });
        
        t2 = clock();
        
        //res = _nn.test(test_data, test_labels2);
        std::cout << "PSOBP results: " << res.num_success << "/" << res.num_total << " Time: " << (t2 - t1)/CLOCKS_PER_SEC << std::endl;
    }
    
    tiny_dnn::network<tiny_dnn::sequential> get_nn(){
        return _nn;
    }
};

#endif /* pso_bp_hpp */
