#ifndef pso_h
#define pso_h

#include <vector>
#include <algorithm>
#include <iostream>

#include "topology.h"
#include "../tiny-dnn/tiny_dnn/tiny_dnn.h"

using tiny_dnn::vec_t;

class SwarmParameters{
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
    
    double threshold;
    int maxPSOEpochs;
    int minPSOEpochs;
};

template <class FitnessFunction>
class Particle {
public:
    vec_t pos;
    vec_t speed;
    double currentFitnVal;
    
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
    
    void InitParticle(FitnessFunction fitnessFunction, double currentVelocityRatio, double localVelocityRatio, double globalVelocityRatio, size_t dimension, double minPosValue, double maxPosValue, double minVelValue, double maxVelValue, double deltaTau, unsigned &seed )
    {
        this->currentVelocityRatio = currentVelocityRatio;
        this->localVelocityRatio = localVelocityRatio;
        this->globalVelocityRatio = globalVelocityRatio;
        this->minPosValue = minPosValue;
        this->maxPosValue = maxPosValue;
        this->minVelValue = minVelValue;
        this->maxVelValue = maxVelValue;
        this->_fitnessFunction = fitnessFunction;
        this->deltaTau = deltaTau;
        
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
            speed[i] = (currentVelocityRatio * speed[i]) + (localVelocityRatio * rnd_localBestPosition[i] * (localBestPos[i] - pos[i])) + (globalVelocityRatio * rnd_globalBestPosition[i] * (globalBestPos[i] - pos[i]));
            
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
        
        currentFitnVal = _fitnessFunction.f(pos);
        if(currentFitnVal < localBestFitnVal)
        {
            localBestFitnVal = currentFitnVal;
            localBestPos = pos;
        }
    }
};

template <class FitnessFunction>
class Swarm {
    
    std::vector<Particle<FitnessFunction>> population;
    
    Topology topologyNB;
    double globalBestFitnVal;
    vec_t globalBestPos;
    
    SwarmParameters sParams;
public:
    
    Swarm(FitnessFunction _fitnessFunction, SwarmParameters _sp)
    {
        sParams = _sp;
        population.resize(sParams.populationSize);
        globalBestFitnVal = -1;
        
        srand(sParams.seed);
        
        topologyNB.init(sParams.topology, sParams.populationSize);
        
        for(int i = 0; i < sParams.populationSize; i++)
        {
            unsigned seed = rand();
            population[i].InitParticle(_fitnessFunction, sParams.currentVelocityRatio, sParams.localVelocityRatio, sParams.globalVelocityRatio, sParams.dimension, sParams.minPosValue, sParams.maxPosValue, sParams.minVelValue, sParams.maxVelValue,
                                       sParams.deltaTau, seed);
            
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
        
        #ifdef LOG
        std::cout << "\nEpoch: " << 0 << " Best fitn val: " << globalBestFitnVal << std::endl;
        #endif
    }
    
    void optimize()
    {
        double prev_variance = 0;
        double cnt_variance = sParams.threshold + 0.1;
        int i = 0;
        while(((std::fabs(cnt_variance - prev_variance) > sParams.threshold) && (i < sParams.maxPSOEpochs)) || (i < sParams.minPSOEpochs))
        {
            double f = 1;
            double f_avg = 0;
            prev_variance = cnt_variance;
            cnt_variance = 0;
            
            //double savedGlobalBestFitnVal = globalBestFitnVal;
            //vec_t savedGlobalBestPos = globalBestPos;
        
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
                
                population[i].NextIteration(topologyNB.getBestPos(i), topologyNB.getBestFitnVal(i), seed);
                
                #ifdef LOG
                std::cout << i << ": " << population[i].currentFitnVal << " Local best fitness: " << population[i].localBestFitnVal << std::endl;
                #endif
            }
            
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
    
    double getSolution(vec_t &v)
    {
        v = globalBestPos;
        return globalBestFitnVal;
    }
    
};

#endif /* pso_h */
