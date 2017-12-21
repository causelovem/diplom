#ifndef fitnessFunctions_h
#define fitnessFunctions_h

#ifndef NO_OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 4
#endif

/*************** Fitness Functions *********/
template <class Activate>
class MyFitnessFunctionMSE{
public:
    std::vector<vec_t> *_train_data;
    std::vector<vec_t> *_train_labels;
    std::vector<int> *_structure;
    Activate _activate_function;
    
    vec_t feed_forward(const vec_t &codeOfNN,const tiny_dnn::vec_t &input){
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
    double f(const vec_t &v){
        double mse = 0.0;
        
        for(size_t i =0; i < _train_data->size(); i++)
        {
            vec_t res = feed_forward(v, (*_train_data)[i]);
            for(int j = 0; j < res.size(); j++)
                mse += (res[j] - (*_train_labels)[i][j]) * (res[j] - (*_train_labels)[i][j]) ;
        }
        
        return mse / _train_labels->size();
        
    }
};

template <class Activate>
class MyFitnessFunctionCE{
public:
    std::vector<vec_t> *_train_data;
    std::vector<vec_t> *_train_labels;
    std::vector<int> *_structure;
    Activate _activate_function;
    
    vec_t feed_forward(const vec_t &codeOfNN,const tiny_dnn::vec_t &input){
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
    double f(const vec_t &v){
        double ce = 0.0;
        cross_entropy _ce;
        
        for(size_t i =0; i < _train_data->size(); i++)
        {
            vec_t res = feed_forward(v, (*_train_data)[i]);
            ce += _ce.f(res, (*_train_labels)[i]);
        }
        
        return ce / _train_data->size();
        
    }
};

class Schwefel{
public:
    double f(const vec_t &v){
        double res = 0.0;
        for(int i = 0; i < v.size(); i++)
            res += v[i] * sin(sqrt(std::fabs(v[i])));
        return (-1)*res/v.size();
    }
};

class Rastrigin{
public:
    double f(const vec_t &v){
        double res = 0.0;
        for(int i = 0; i < v.size(); i++)
            res += v[i]*v[i] - 10*cos(6.28*v[i]) + 10;
        return res;
    }
};
/*******************************************/

#endif /* fitnessFunctions_h */
