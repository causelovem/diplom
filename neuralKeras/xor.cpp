#include "tiny_dnn/tiny_dnn.h"

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
    // std::vector<vec_t> train_numbers{ {0, 0}, {0, 1}, {1, 0}, {1, 1} };

    std::vector<vec_t> train_numbers;

    vec_t tmp;
    tmp.push_back(0);
    tmp.push_back(0);

    train_numbers.push_back(tmp);
    tmp.clear();

    tmp.push_back(0);
    tmp.push_back(1);

    train_numbers.push_back(tmp);
    tmp.clear();

    tmp.push_back(1);
    tmp.push_back(0);

    train_numbers.push_back(tmp);
    tmp.clear();

    tmp.push_back(1);
    tmp.push_back(1);

    train_numbers.push_back(tmp);
    tmp.clear();


    adagrad optimizer; // use gradient_descent?
    net.train<mse>(optimizer, train_numbers, train_labels, 4, 100); // batch size 4, 1000 epochs

    for (auto& tn : train_numbers)
    {
        auto res_label = net.predict_label(tn);
        auto res = net.predict(tn);
        std::cout << "In: (" << tn[0] << "," << tn[1] << ") Prediction: " << res_label << std::endl;
    }
    
    return 0;
}
