#ifndef apso_h
#define apso_h

#include <vector>
#include <algorithm>
#include <iostream>

#include "../tiny-dnn/tiny_dnn/tiny_dnn.h"

using tiny_dnn::vec_t;

class AdaptiveSwarmParameters{
public:
    unsigned seed;
    
    double inirtiaWeight;
    double maxInirtiaWeight;
    double minInirtiaWeight;
    double localVelocityRatio;
    double globalVelocityRatio;
    size_t populationSize;
    size_t dimension;
    double deltaTau;
    int maxPSOEpochs;
};

template <class FitnessFunction>
class AdaptiveParticle {
public:
    vec_t pos;
    vec_t speed;
    double currentFitnVal;
    
    double inirtiaWeight;
    double maxInirtiaWeight;
    double minInirtiaWeight;
    double localVelocityRatio;
    double globalVelocityRatio;
    double deltaTau;
    double maxPSOEpochs;
    double curPSOEpoch;
    double dimension;
    
    double localBestFitnVal;
    vec_t localBestPos;
    
    FitnessFunction fitnessFunction;
    
    void InitParticle(FitnessFunction fitnessFunction, double dimension, double inirtiaWeight, double maxInirtiaWeight, double minInirtiaWeight, double localVelocityRatio, double globalVelocityRatio, double deltaTau, double maxPSOEpochs, unsigned &seed)
    {
        this->inirtiaWeight = inirtiaWeight;
        this->maxInirtiaWeight = maxInirtiaWeight;
        this->minInirtiaWeight = minInirtiaWeight;
        this->localVelocityRatio = localVelocityRatio;
        this->globalVelocityRatio = globalVelocityRatio;
        this->deltaTau = deltaTau;
        this->maxPSOEpochs = maxPSOEpochs;
        this->curPSOEpoch = 0;
        this->dimension = dimension;
        this->fitnessFunction = fitnessFunction;
        
        pos.resize(dimension);
        speed.resize(dimension);
        
        double a = -1;
        double b = 1;
        std::generate(pos.begin(), pos.end(),[a, b, &seed]{return double(rand_r(&seed)) / RAND_MAX * (b - a) + a;});
        
        a = -0.3;
        b = 0.3;
        std::generate(speed.begin(), speed.end(),[a, b, &seed]{return double(rand_r(&seed)) / RAND_MAX * (b - a) + a;});
        
        currentFitnVal = fitnessFunction.f(pos);
        localBestFitnVal = currentFitnVal;
        localBestPos = pos;
    }
    
    void NextIteration(const vec_t &globalBestPos, double globalBestFitnVal, unsigned &seed)
    {
        double a = 0;
        double b = 1;
        
        vec_t rnd_localBestPosition(pos.size());
        vec_t rnd_globalBestPosition(pos.size());
        std::generate(rnd_localBestPosition.begin(), rnd_localBestPosition.end(),[a, b, &seed]{return double(rand_r(&seed)) / RAND_MAX * (b - a) + a;});
        std::generate(rnd_globalBestPosition.begin(), rnd_globalBestPosition.end(),[a, b, &seed]{return double(rand_r(&seed)) / RAND_MAX * (b - a) + a;});
        
        for(int i = 0; i < speed.size(); i++)
        {
            inirtiaWeight = minInirtiaWeight - curPSOEpoch*(maxInirtiaWeight - minInirtiaWeight)/maxPSOEpochs;
            
            speed[i] = (inirtiaWeight * speed[i]) + (localVelocityRatio * rnd_localBestPosition[i] * (localBestPos[i] - pos[i])) + (globalVelocityRatio * rnd_globalBestPosition[i] * (globalBestPos[i] - pos[i]));
            
            pos[i] += deltaTau * speed[i];
        }
        curPSOEpoch++;
        
        currentFitnVal = fitnessFunction.f(pos);
        if(currentFitnVal < localBestFitnVal)
        {
            localBestFitnVal = currentFitnVal;
            localBestPos = pos;
        }
    }
};

template <class FitnessFunction>
class AdaptiveSwarm {
    
    std::vector<AdaptiveParticle<FitnessFunction>> population;
    
    double globalBestFitnVal;
    vec_t globalBestPos;
    
    AdaptiveSwarmParameters sParams;
public:
    
    AdaptiveSwarm(FitnessFunction _fitnessFunction, AdaptiveSwarmParameters _sp)
    {
        sParams = _sp;
        population.resize(sParams.populationSize);
        globalBestFitnVal = -1;
        
        for(int i = 0; i < sParams.populationSize; i++)
        {
            unsigned seed = rand_r(&(sParams.seed));
            population[i].InitParticle(_fitnessFunction, sParams.dimension, sParams.inirtiaWeight, sParams.maxInirtiaWeight, sParams.minInirtiaWeight, sParams.localVelocityRatio, sParams.globalVelocityRatio, sParams.deltaTau, sParams.maxPSOEpochs, seed);
            
            if((population[i].localBestFitnVal < globalBestFitnVal) || (globalBestFitnVal == -1))
            {
                globalBestFitnVal = population[i].localBestFitnVal;
                globalBestPos = population[i].localBestPos;
            }
        }
        
#ifdef LOG
        std::cout << "\nEpoch: " << 0 << " Best fitn val: " << globalBestFitnVal << std::endl;
#endif
    }
    
    void optimize()
    {
        int i = 0;
        while(i < sParams.maxPSOEpochs)
        {
            
            double savedGlobalBestFitnVal = globalBestFitnVal;
            vec_t savedGlobalBestPos = globalBestPos;
            
            for(int i = 0; i < population.size(); i++)
            {
                if(population[i].localBestFitnVal < globalBestFitnVal)
                {
                    globalBestFitnVal = population[i].localBestFitnVal;
                    globalBestPos = population[i].localBestPos;
                }
                
                unsigned seed = rand();
                population[i].NextIteration(savedGlobalBestPos, savedGlobalBestFitnVal, seed);
            }
            i++;
#ifdef LOG
            std::cout << "\nEpoch: " << i << " Best fitn val: " << globalBestFitnVal << std::endl;
#endif
        }
    }
    
    double getSolution(vec_t &v)
    {
        v = globalBestPos;
        return globalBestFitnVal;
    }
    
};

#endif /* apso_h */
