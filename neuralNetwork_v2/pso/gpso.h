#ifndef gpso_h
#define gpso_h

#include <vector>
#include <algorithm>
#include <iostream>

#include "topology.h"
#include "../tiny-dnn/tiny_dnn/tiny_dnn.h"

using tiny_dnn::vec_t;

class GlobalSwarmParameters{
public:
    int seed;
    char *topology;
    double currentVelocityRatio;
    double localVelocityRatio;
    double globalVelocityRatio;
    size_t populationSize;
    size_t dimension;
    double minPosValue;
    double maxPosValue;
    double minVelValue;
    double maxVelValue;
    double deltaTau;
    double max_gen1;
    double w1;
    double k;
    
    double threshold;
    int maxPSOEpochs;
    int minPSOEpochs;
};

template <class FitnessFunction>
class GlobalParticle {
public:
    vec_t pos;
    vec_t speed;
    double currentFitnVal;
    
    int cnt_generation;
    int max_gen1;
    double w0, w1, k;
    double currentVelocityRatio;
    double localVelocityRatio;
    double globalVelocityRatio;
    double minPosValue;
    double maxPosValue;
    double minVelValue;
    double maxVelValue;
    double deltaTau;
    
    double localBestFitnVal;
    vec_t localBestPos;
    
    FitnessFunction _fitnessFunction;
    
    void InitParticle(FitnessFunction fitnessFunction, double currentVelocityRatio, double localVelocityRatio, double globalVelocityRatio, size_t dimension, double minPosValue, double maxPosValue, double minVelValue, double maxVelValue, double deltaTau, int _max_gen1, double w1, double k, unsigned &seed ){
        this->currentVelocityRatio = currentVelocityRatio;
        this->localVelocityRatio = localVelocityRatio;
        this->globalVelocityRatio = globalVelocityRatio;
        this->minPosValue = minPosValue;
        this->maxPosValue = maxPosValue;
        this->minVelValue = minVelValue;
        this->maxVelValue = maxVelValue;
        this->_fitnessFunction = fitnessFunction;
        this->deltaTau = deltaTau;
        this->cnt_generation = 1;
        this->max_gen1 = _max_gen1;
        this->k = k;
        this->w0 = currentVelocityRatio;
        this->w1 = w1;
        
        pos.resize(dimension);
        speed.resize(dimension);
        
        double a = minPosValue;
        double b = maxPosValue;
        std::generate(pos.begin(), pos.end(),[a, b, &seed]{return double(rand_r(&seed)) / RAND_MAX * (b - a) + a;});
        
        a = minVelValue;
        b = maxVelValue;
        std::generate(speed.begin(), speed.end(),[a, b, &seed]{return double(rand_r(&seed)) / RAND_MAX * (b - a) + a;});
        
        currentFitnVal = _fitnessFunction.f(pos);
        localBestFitnVal = currentFitnVal;
        localBestPos = pos;
    }
    
    void NextIteration(const vec_t &globalBestPos, double globalBestFitnVal, const vec_t &improvementFactor, unsigned &seed){
        double a = 0;
        double b = 1;
        
        vec_t rnd_localBestPosition(pos.size());
        vec_t rnd_globalBestPosition(pos.size());
        vec_t rnd_ImprovementFactorPosition(pos.size());
        std::generate(rnd_localBestPosition.begin(), rnd_localBestPosition.end(),[a, b, &seed]{return double(rand_r(&seed)) / RAND_MAX * (b - a) + a;});
        std::generate(rnd_globalBestPosition.begin(), rnd_globalBestPosition.end(),[a, b, &seed]{return double(rand_r(&seed)) / RAND_MAX * (b - a) + a;});
        std::generate(rnd_ImprovementFactorPosition.begin(), rnd_ImprovementFactorPosition.end(),[a, b, &seed]{return double(rand_r(&seed)) / RAND_MAX * (b - a) + a;});
        
        double c3 = (localVelocityRatio + globalVelocityRatio) / 2;
        
        for(int i = 0; i < speed.size(); i++)
        {
            speed[i] = (currentVelocityRatio * speed[i]) + (localVelocityRatio * rnd_localBestPosition[i] * (localBestPos[i] - pos[i])) + (globalVelocityRatio * rnd_globalBestPosition[i] * (globalBestPos[i] - pos[i])) + (c3 * rnd_ImprovementFactorPosition[i] * (improvementFactor[i] - pos[i]));
            
            if(cnt_generation < max_gen1)
                currentVelocityRatio = w0 - (w1/max_gen1)*cnt_generation;
            else
                currentVelocityRatio = (w0 - w1) * exp((max_gen1 - cnt_generation)/k);
            
            pos[i] += deltaTau * speed[i];
            
            if(pos[i] > maxPosValue)
                pos[i] = maxPosValue;
            if(pos[i] < minPosValue)
                pos[i] = minPosValue;
            
            if(speed[i] > maxVelValue)
                speed[i] = maxVelValue;
            if(speed[i] < minVelValue)
                speed[i] = minVelValue;
        }
        
        cnt_generation++;
        
        currentFitnVal = _fitnessFunction.f(pos);
        if(currentFitnVal < localBestFitnVal)
        {
            localBestFitnVal = currentFitnVal;
            localBestPos = pos;
        }
    }
};

template <class FitnessFunction>
class GlobalSwarm {
    
    std::vector<GlobalParticle<FitnessFunction>> population;
    
    Topology topologyNB;
    double globalBestFitnVal;
    vec_t globalBestPos;
    
    std::vector<vec_t> bestPos;
    
    GlobalSwarmParameters sParams;
    
public:
    
    GlobalSwarm(FitnessFunction _fitnessFunction, GlobalSwarmParameters _sp){
        sParams = _sp;
        population.resize(sParams.populationSize);
        globalBestFitnVal = -1;
        
        srand(sParams.seed);
        
        topologyNB.init(sParams.topology, sParams.populationSize);
        
        for(int i = 0; i < sParams.populationSize; i++)
        {
            unsigned seed = rand();
            population[i].InitParticle(_fitnessFunction, sParams.currentVelocityRatio, sParams.localVelocityRatio, sParams.globalVelocityRatio, sParams.dimension, sParams.minPosValue, sParams.maxPosValue, sParams.minVelValue, sParams.maxVelValue,
                                       sParams.deltaTau, sParams.max_gen1, sParams.w1, sParams.k, seed);
            
            if((population[i].localBestFitnVal < globalBestFitnVal) || (globalBestFitnVal == -1))
            {
                globalBestFitnVal = population[i].localBestFitnVal;
                globalBestPos = population[i].localBestPos;
            }
            
            if((population[i].localBestFitnVal < topologyNB.getBestFitnVal(i)) || (topologyNB.getBestFitnVal(i)  == -1))
                topologyNB.setBestFitn(i, population[i].localBestFitnVal, population[i].localBestPos);
            
            #ifdef LOG
            std::cout << i << ": " << population[i].currentFitnVal << std::endl;
            #endif
        }
        
        bestPos.push_back(globalBestPos);
        
        #ifdef LOG
        std::cout << "\nEpoch: " << 0 << " Best fitn val: " << globalBestFitnVal << std::endl;
        #endif
        
    }
    
    void optimize(){
        double prev_variance = 0;
        double cnt_variance = sParams.threshold + 0.1;
        int i = 0;
        while(((std::fabs(cnt_variance - prev_variance) > sParams.threshold) && (i < sParams.maxPSOEpochs)) || (i < sParams.minPSOEpochs))
        {
            double f = 1;
            double f_avg = 0;
            prev_variance = cnt_variance;
            cnt_variance = 0;
            
            std::vector<std::pair<vec_t, double>> saved;
            for(int i = 0; i < population.size(); i++)
                saved.push_back(std::pair<vec_t, double>(topologyNB.getBestPos(i), topologyNB.getBestFitnVal(i)));
                
            vec_t improve;
            improve.resize(globalBestPos.size());
            unsigned chIter = double(rand()) / RAND_MAX * (bestPos.size() - 1);
            for(int i = 0; i < improve.size(); i++)
            {
                unsigned chPath = double(rand()) / RAND_MAX * (globalBestPos.size() - 1);
                while(chPath == i)
                    chPath = double(rand()) / RAND_MAX * (globalBestPos.size() - 1);
                improve[i] = bestPos[chIter][chPath];
            }
            
            unsigned seed = rand();
            for(int i = 0; i < population.size(); i++)
            {
                f_avg += population[i].currentFitnVal;
                
                if(population[i].localBestFitnVal < globalBestFitnVal)
                {
                    globalBestFitnVal = population[i].localBestFitnVal;
                    globalBestPos = population[i].localBestPos;
                }
                
                if(population[i].localBestFitnVal < topologyNB.getBestFitnVal(i))
                    topologyNB.setBestFitn(i, population[i].localBestFitnVal, population[i].localBestPos);
                
                population[i].NextIteration(saved[i].first, topologyNB.getBestFitnVal(i), improve, seed);
                
                #ifdef LOG
                std::cout << i << ": " << population[i].currentFitnVal << " Local best fitness: " << population[i].localBestFitnVal << std::endl;
                #endif
            }
            
            bestPos.push_back(globalBestPos);
            
            f_avg /= population.size();
            
            for(int i = 0; i < population.size(); i++)
            {
                if(std::fabs(population[i].currentFitnVal - f_avg) > f)
                    f = std::fabs(population[i].currentFitnVal - f_avg);
                cnt_variance += (population[i].currentFitnVal - f_avg) * (population[i].currentFitnVal - f_avg);
            }
            
            cnt_variance *= 1/(f*f);
            i++;
            
            #ifdef LOG
            std::cout << "\nEpoch: " << i << " Best fitn val: " << globalBestFitnVal << std::endl;
            #endif
        }
        
        #ifdef LOG
        std::cout << "diff variance: " << std::fabs(cnt_variance - prev_variance) <<std::endl;
        #endif
    }
    
    double getSolution(vec_t &v){
        v = globalBestPos;
        return globalBestFitnVal;
    }
    
};

#endif /* pso_h */
