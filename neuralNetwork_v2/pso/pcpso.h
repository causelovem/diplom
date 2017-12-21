#ifndef pcpso_h
#define pcpso_h
#ifndef NO_MPI

#include <vector>
#include <algorithm>
#include <iostream>
#include "mpi.h"

#include "../tiny-dnn/tiny_dnn/tiny_dnn.h"

using tiny_dnn::vec_t;
using std::vector;

struct psswarm{
    vector<vec_t> speeds;
    vector<vec_t> positions;
    vector<vec_t> lbestpos;
    vec_t gbestpos;
};

class ParallelCooperativeSwarmParameters {
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
    
    int argc;
    char** argv;
    int rank;
    int groupSize;
};

template <class FitnessFunction>
class ParallelCooperativeSwarm {
    vector<sswarm> _sw;
    FitnessFunction ff;
    ParallelCooperativeSwarmParameters sParams;
    
    vec_t bestpos;
    double bestfit;
    
    void concatenate2(const vec_t &src, vec_t &result,const vec_t &part, double partNum){
        result = src;
        for(int i = 0; i < part.size(); i++)
            result[i + partNum*sParams.dimension / sParams.swarmsNum] = part[i];
    }
    
    void init_best_pos() {
        for (int i = 0; i < sParams.populationSize; i++) {
            vec_t tmp;
            for(int j = 0; j < sParams.swarmsNum; j++)
                tmp.insert(tmp.end(), _sw[j].positions[i].begin(), _sw[j].positions[i].end());
            if(bestpos.size() == 0) {
                bestpos = tmp;
                bestfit = ff.f(tmp);
            }
            else{
                double tmpfit = ff.f(tmp);
                if(bestfit > tmpfit) {
                    bestpos = tmp;
                    bestfit = tmpfit;
                }
            }
        }
        
        for(int i = 0; i < _sw.size(); i++) {
            _sw[i].gbestpos.clear();
            _sw[i].gbestpos.insert(_sw[i].gbestpos.end(), bestpos.begin() + i*_sw[i].positions[0].size(), bestpos.begin() + (i+1)*_sw[i].positions[0].size());
        }
    }
    
    void waitForCalculations(){
        while(1){
            vec_t buf(2 + 2 * sParams.dimension);
            vec_t t1, t2;
            
            MPI_Recv(buf.data(), buf.size(), MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(buf[0] == -1)
                break;
            
            t1.insert(t1.end(), buf.begin() + 2, buf.begin() + 2 + sParams.dimension);
            t2.insert(t2.end(), buf.begin() + 2 + sParams.dimension, buf.end());
            vec_t res;
            res.push_back(buf[0]);
            res.push_back(buf[1]);
            res.push_back(ff.f(t1));
            res.push_back(ff.f(t2));
            
            MPI_Send(res.data(), res.size(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        exit(0);
    }
    
public:
    
    ParallelCooperativeSwarm(FitnessFunction fff, ParallelCooperativeSwarmParameters _sp) {
        ff = fff;
        sParams = _sp;

        MPI_Init(&(sParams.argc),&(sParams.argv));
        MPI_Comm_rank(MPI_COMM_WORLD,&(sParams.rank));
        MPI_Comm_size(MPI_COMM_WORLD,&(sParams.groupSize));
        
        if(sParams.rank != 0)
            waitForCalculations();
        
        _sw.resize(_sp.swarmsNum);
        for(int i = 0; i < _sw.size(); i++){
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
    
    void optimize() {
        for(int i = 0; i < sParams.maxPSOEpochs; i++) {
            
            sParams.inirtiaWeight = sParams.maxInirtiaWeight - (sParams.maxInirtiaWeight - sParams.minInirtiaWeight) * i / sParams.maxPSOEpochs;
            
            // Передача на slave узлы частицы для вычислений
            vector<std::pair<vec_t, double>> m(_sw.size(), std::pair<vec_t, double>(bestpos, bestfit));

            int nextNode = 1;
            for(int j = 0; j < _sw.size(); j++)
                for(int k = 0; k < sParams.populationSize; k++) {
                    vec_t t1, t2;
                    concatenate2(bestpos, t1, _sw[j].positions[k], j);
                    concatenate2(bestpos, t2, _sw[j].lbestpos[k], j);
                    
                    // Send j, k, t1, t2;
                    vec_t buf;
                    buf.push_back(j);
                    buf.push_back(k);
                    buf.insert(buf.end(), t1.begin(), t1.end());
                    buf.insert(buf.end(), t2.begin(), t2.end());
                    MPI_Request req;
                    MPI_Isend(buf.data(), buf.size(), MPI_FLOAT, nextNode, 0, MPI_COMM_WORLD, &req);
                    nextNode++;
                    
                    if(nextNode == sParams.groupSize){
                        for(int l = 1; l < nextNode; l++) {
                            
                            //Recv j, k, r1, r2;
                            vec_t buf(4);
                            MPI_Recv(buf.data(), buf.size(), MPI_FLOAT, l, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            
                            int a = buf[0];
                            int b = buf[1];
                            double res1 = buf[2];
                            double res2 = buf[3];
                            
                            //std::cout << "Gotcha: " << a << " " << b << " " << res1 << " " << res2 << std::endl;
                            
                            if(res1 < res2)
                                _sw[a].lbestpos[b] = _sw[a].positions[b];
                            if(res1 < m[a].second){
                                vec_t t1;
                                concatenate2(bestpos, t1, _sw[a].positions[b], a);
                                m[a].first = t1;
                                m[a].second = res1;
                                _sw[a].gbestpos = _sw[a].positions[b];
                            }
                        }
                        nextNode = 1;
                    }
                }
            
            if(nextNode != 1){
                for(int l = 1; l < nextNode; l++) {
                    
                    //Recv j, k, r1, r2;
                    vec_t buf(4);
                    MPI_Recv(buf.data(), buf.size(), MPI_FLOAT, l, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    int a = buf[0];
                    int b = buf[1];
                    double res1 = buf[2];
                    double res2 = buf[3];
                    
                    //std::cout << "Gotcha: " << a << " " << b << " " << res1 << " " << res2 << std::endl;
                    
                    if(res1 < res2)
                        _sw[a].lbestpos[b] = _sw[a].positions[b];
                    if(res1 < m[a].second){
                        vec_t t1;
                        concatenate2(bestpos, t1, _sw[a].positions[b], a);
                        m[a].first = t1;
                        m[a].second = res1;
                        _sw[a].gbestpos = _sw[a].positions[b];
                    }
                }
                nextNode = 1;
            }
            
            // Выбор нового лучшего значения
            for(int j = 0; j < m.size(); j++){
                if(m[j].second < bestfit){
                    bestfit = m[j].second;
                    bestpos = m[j].first;
                }
            }
            
            for(int j = 0; j < _sw.size(); j++)
                for(int k = 0; k < _sw[j].gbestpos.size(); k++)
                    _sw[j].gbestpos[k] = bestpos[k + j*sParams.dimension / sParams.swarmsNum];
            
            if(sParams.rank == 0)
                std::cout << i << " " << "BestFit: " << bestfit << std::endl;
            
            // Обновление скоростей и позиций
            for(int j = 0; j < _sw.size(); j++)
                for(int k = 0; k < sParams.populationSize; k++) {
                    for(int m = 0; m < _sw[j].speeds[k].size(); m++) {
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
            
            //ДОБАВИТЬ стандартное отклонение
        }
        
        vec_t end(2 + 2 * sParams.dimension);
        end[0] = -1;
        for(int i = 1; i < sParams.groupSize; i++)
            MPI_Send(end.data(), end.size(), MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
    }
    
    double getSolution(vec_t &v){
        v = bestpos;
        return ff.f(bestpos);
    }
};

#endif
#endif /* pcpso_h */
