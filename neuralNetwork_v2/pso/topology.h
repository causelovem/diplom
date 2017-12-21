#ifndef topology_hpp
#define topology_hpp

#include <stdio.h>
#include <vector>
#include <set>
#include <string.h>

#include "../tiny-dnn/tiny_dnn/tiny_dnn.h"

using tiny_dnn::vec_t;

class Topology {
    std::vector<std::pair<vec_t, double>> vals;
    std::vector<std::set<int>> structure;
public:
    void init(char *type, size_t num){
        if(strcmp(type, "circle") == 0)
        {
            structure.resize(num);
            vals.resize(num);
            for(int i = 1; i < num-1; i++)
            {
                structure[i].insert(i-1);
                structure[i].insert(i);
                structure[i].insert(i+1);
                vals[i].second = -1;
            }
            structure[0].insert(1);
            structure[0].insert(0);
            structure[0].insert(int(num-1));
            vals[0].second = -1;
            
            structure[num-1].insert(0);
            structure[num-1].insert(int(num-1));
            structure[num-1].insert(int(num-2));
            vals[num-1].second = -1;
        }
        else if(strcmp(type, "tor") == 0)
        {
            
        }
        else if(strcmp(type, "cluster") == 0)
        {
            
        }
        else
        {
            //fulllinking
            structure.resize(num);
            vals.resize(num);
            for(int i = 0; i < num; i++)
            {
                for(int j = 0; j < num; j++)
                    structure[i].insert(j);
                vals[i].second = -1;
            }
        }
    }
    
    void setBestFitn(int i, double val, const vec_t &pos){
        vals[i].second = val;
        vals[i].first = pos;
    }
    
    double getBestFitnVal(int i){
        double fit =  vals[*(structure[i].begin())].second;
        for(auto it : structure[i])
        {
            if(vals[it].second < fit)
                fit = vals[it].second;
        }
        return fit;
    }
    
    const vec_t & getBestPos(int i){
        int n_fit =  *(structure[i].begin());
        for(auto it : structure[i])
        {
            if(vals[it].second < vals[n_fit].second)
                n_fit = it;
        }
        return vals[n_fit].first;
    }
};

#endif /* topology_hpp */
