#ifndef cpso_h
#define cpso_h

#include <vector>
#include <algorithm>
#include <iostream>

#include "../tiny-dnn/tiny_dnn/tiny_dnn.h"

using tiny_dnn::vec_t;
using std::vector;

struct sswarm{
    vector<vec_t> speeds;
    vector<vec_t> positions;
    vector<vec_t> lbestpos;
    vec_t gbestpos;
};

class CooperativeSwarmParameters {
public:
    unsigned seed;
    
    size_t dimension;
    size_t populationSize;
    size_t swarmsNum;
    double localVelocityRatio;
    double globalVelocityRatio;
    double inirtiaWeight;
    double maxInirtiaWeight;
    double minInirtiaWeight;
    double maxPSOEpochs;
    
    double u_bord;
    double l_bord;
};

template <class FitnessFunction>
class CooperativeSwarm {
    vector<sswarm> _sw;
    FitnessFunction ff;
    CooperativeSwarmParameters sParams;
    
    vec_t bestpos;
    double bestfit;
    
    void concatenate2(const vec_t &src, vec_t &result,const vec_t &part, double partNum){
        result = src;
        for(int i = 0; i < part.size(); i++)
            result[i + partNum*sParams.dimension / sParams.swarmsNum] = part[i];
    }
    
    void init_best_pos(){
        for (int i = 0; i < sParams.populationSize; i++) {
            vec_t tmp;
            for(int j = 0; j < sParams.swarmsNum; j++)
                tmp.insert(tmp.end(), _sw[j].positions[i].begin(), _sw[j].positions[i].end());
            if(bestpos.size() == 0)
            {
                bestpos = tmp;
                bestfit = ff.f(tmp);
            }
            else
            {
                double tmpfit = ff.f(tmp);
                if(bestfit > tmpfit)
                {
                    bestpos = tmp;
                    bestfit = tmpfit;
                }
            }
        }
        
        for(int i = 0; i < _sw.size(); i++){
            _sw[i].gbestpos.clear();
            _sw[i].gbestpos.insert(_sw[i].gbestpos.end(), bestpos.begin() + i*_sw[i].positions[0].size(), bestpos.begin() + (i+1)*_sw[i].positions[0].size());
        }
    }
    
public:
    
    CooperativeSwarm(FitnessFunction fff, CooperativeSwarmParameters _sp){
        ff = fff;
        sParams = _sp;
        _sw.resize(_sp.swarmsNum);
        for(int i = 0; i < _sw.size(); i++)
        {
            size_t inSwarmSize = sParams.dimension / sParams.swarmsNum;
            
            _sw[i].speeds.resize(sParams.populationSize, vec_t(inSwarmSize, 0));
            _sw[i].positions.resize(sParams.populationSize, vec_t(inSwarmSize, 0));
            _sw[i].lbestpos.resize(sParams.populationSize, vec_t(inSwarmSize, 0));
            _sw[i].gbestpos.resize(inSwarmSize, 0);
            
            double a = sParams.l_bord * 0.5;
            double b = sParams.u_bord * 0.5;
            for(int j = 0; j < _sw[i].speeds.size(); j++)
            {
                unsigned seed = rand_r(&sParams.seed);
                std::generate(_sw[i].speeds[j].begin(), _sw[i].speeds[j].end(),[a, b, &seed]{return double(rand_r(&seed)) / RAND_MAX * (b - a) + a;});
            }
            
            a = sParams.l_bord;
            b = sParams.u_bord;
            for(int j = 0; j < _sw[i].positions.size(); j++)
            {
                unsigned seed = rand_r(&sParams.seed);
                std::generate(_sw[i].positions[j].begin(), _sw[i].positions[j].end(),[a, b, &seed]{return double(rand_r(&seed)) / RAND_MAX * (b - a) + a;});
                _sw[i].lbestpos[j] = _sw[i].positions[j];
            }
        }
        
        init_best_pos();
    }
    
    void optimize(){
        for(int i = 0; i < sParams.maxPSOEpochs; i++){
            
            sParams.inirtiaWeight = sParams.maxInirtiaWeight - (sParams.maxInirtiaWeight - sParams.minInirtiaWeight) * i / sParams.maxPSOEpochs;
            
            vector<std::pair<vec_t, double>> m(_sw.size(), std::pair<vec_t, double>(bestpos, bestfit));
            
            for(int j = 0; j < _sw.size(); j++){
                
                for(int k = 0; k < sParams.populationSize; k++){
                    vec_t t1, t2;
                    concatenate2(m[j].first, t1, _sw[j].positions[k], j);
                    concatenate2(m[j].first, t2, _sw[j].lbestpos[k], j);
                    
                    double res1 = ff.f(t1);
                    double res2 = ff.f(t2);
                    
                    if(res1 < res2)
                        _sw[j].lbestpos[k] = _sw[j].positions[k];
                    if(res1 < m[j].second){
                        m[j].first = t1;
                        m[j].second = res1;
                        _sw[j].gbestpos = _sw[j].positions[k];
                    }
                }
                
                for(int k = 0; k < sParams.populationSize; k++){
                    for(int m = 0; m < _sw[j].speeds[k].size(); m++){
                        double r1 = double(rand_r(&sParams.seed)) / RAND_MAX;
                        double r2 = double(rand_r(&sParams.seed)) / RAND_MAX;
                        _sw[j].speeds[k][m] = sParams.inirtiaWeight * _sw[j].speeds[k][m] + sParams.localVelocityRatio * r1 * (_sw[j].positions[k][m] - _sw[j].lbestpos[k][m]) + sParams.globalVelocityRatio * r2 * (_sw[j].positions[k][m] - _sw[j].gbestpos[m]);
                        
                        if((_sw[j].speeds[k][m] > sParams.u_bord) || (_sw[j].speeds[k][m] < sParams.l_bord))
                            _sw[j].speeds[k][m] *= -1;
                        
                        _sw[j].positions[k][m] += _sw[j].speeds[k][m];
                        
                        if((_sw[j].positions[k][m] > sParams.u_bord) || (_sw[j].positions[k][m] < sParams.l_bord))
                            _sw[j].positions[k][m] = double(rand_r(&sParams.seed)) / RAND_MAX * (sParams.u_bord - sParams.l_bord) + sParams.l_bord;
                    }
                }
                
                /*for(int k = 0; k < sParams.populationSize; k++)
                {
                    for(int m = 0; m < _sw[j]; m++)
                }*/
            }
            
            for(int j = 0; j < m.size(); j++){
                if(m[j].second < bestfit){
                    bestfit = m[j].second;
                    bestpos = m[j].first;
                }
            }
            
            for(int j = 0; j < _sw.size(); j++){
                for(int k = 0; k < _sw[j].gbestpos.size(); k++){
                    _sw[j].gbestpos[k] = bestpos[k + j*sParams.dimension / sParams.swarmsNum];
                }
            }
            
            std::cout << "BestFit: " << ff.f(bestpos) << std::endl;
        }
    }
    
    double getSolution(vec_t &v){
        v = bestpos;
        return ff.f(bestpos);
    }
};

#endif /* cpso_h */
