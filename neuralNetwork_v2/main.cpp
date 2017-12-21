#include <iostream>

#ifdef LOMONOSOV
#define CNN_NO_SERIALIZATION
#define CNN_SINGLE_THREAD
#define CNN_DEFAULT_MOVE_CONSTRUCTOR_UNAVAILABLE
#define CNN_DEFAULT_ASSIGNMENT_OPERATOR_UNAVAILABLE
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"
#include "tiny_dnn/tiny_dnn.h"
#pragma clang pop

#include "examples.h"
#include <fstream>

// int main(int argc, char **argv){
//     srand(unsigned(time(NULL)));
    
//     Letters(argc, argv);
    
//     return 0;
// }

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
network<sequential> construct_mlp() 
{
    //auto mynet = make_mlp<tan_h>({ 2, 8, 2 });
    auto mynet = make_mlp<relu>({ 2, 8, 2 });
    assert(mynet.in_data_size() == 2);
    assert(mynet.out_data_size() == 2);
    return mynet;
}

int main(int argc, char** argv)
{
    auto net = construct_mlp();

    std::vector<label_t> train_labels {0, 1, 1, 0};
    std::vector<vec_t> train_numbers{ {0, 0}, {0, 1}, {1, 0}, {1, 1} };

    int epoch = 0;
    // timer t;
    adagrad optimizer; // use gradient_descent?
    net.train<mse>(optimizer, train_numbers, train_labels, 4, 100,
    	[&](){
           // t.elapsed();
           // t.reset();
         },
	      // called for each epoch
	     [&](){
	       result res = net.test(train_numbers, train_labels);
	       cout << res.num_success << "/" << res.num_total << endl;
	       // std::ofstream ofs (("epoch_"+to_string(epoch++)).c_str());
	       std::cout << net;
         }); // batch size 4, 1000 epochs

    for (auto& tn : train_numbers)
    {
        auto res_label = net.predict_label(tn);
        auto res = net.predict(tn);
        std::cout << "In: (" << tn[0] << "," << tn[1] << ") Prediction: " << res_label << std::endl;
    }
    // std::cin.get();
    return 0;
}
