#include "tiny_dnn/tiny_dnn.h"
#include <fstream>

using namespace std;
using namespace tiny_dnn;
using namespace tiny_dnn::activation;

// network<sequential> construct_mlp() 
// {
//     //auto mynet = make_mlp<tan_h>({ 2, 8, 2 });
//     auto mynet = make_mlp<relu>({ 2, 8, 2 });
//     assert(mynet.in_data_size() == 2);
//     assert(mynet.out_data_size() == 2);
//     return mynet;
// }

int main(int argc, char** argv)
{
    /*<numOfFiles> <matrixFiles> <mappingFiles>*/
    long unsigned int matrixDim = 8;
    int numOfFiles = atoi(argv[1]);

    std::vector<vec_t> inData;
    for (int k = 2; k < numOfFiles + 2; k++)
    {
        ifstream matrixFile(argv[k]);
        vec_t matrixVec;

        int tmp = 0;
        for (int i = 0; i < matrixDim; i++)
            for (int j = 0; j < matrixDim; j++)
            {
                matrixFile >> tmp;
                matrixVec.push_back(tmp);
            }

        inData.push_back(matrixVec);
        matrixVec.clear();

        matrixFile.close();
    }

    cout << inData.size() << endl;

    std::vector<vec_t> desData;
    for (int k = numOfFiles + 2; k < argc; k++)
    {
        ifstream mappingFile(argv[k]);
        vec_t mappingVec;

        int tmp = 0;
        for (int i = 0; i < matrixDim; i++)
            for (int j = 0; j < 4; j++)
            {
                mappingFile >> tmp;
                mappingVec.push_back(tmp);
            }

        desData.push_back(mappingVec);
        mappingVec.clear();

        mappingFile.close();
    }

    cout << desData.size() << endl;

    network<sequential> net = make_mlp<sigmoid>({matrixDim * matrixDim, matrixDim, matrixDim * 4});
    // auto mynet = make_mlp<relu>({ 2, 8, 2 });
    cout << "!!!" << endl;
    // return 0;
    // auto net = make_mlp<sigmoid>({512 * 512, 512, 512 * 4});
    // auto net = construct_mlp();
    // std::vector<label_t> train_labels {0, 1, 1, 0};
    // std::vector<vec_t> train_numbers{ {0, 0}, {0, 1}, {1, 0}, {1, 1} };

    // cout << train_labels << endl;
    // cout << train_numbers << endl;

    int epo = 0;
    timer t;
    adagrad optimizer; // use gradient_descent?
    net.fit<mse>(optimizer, inData, desData, 4, 200,
    // called for each mini-batch
         [&](){
           // cout << t.elapsed() << endl;
           // t.restart();
           // cout << "???" << endl;
         },
         // called for each epoch
         [&](){
           // result res = net.test(inData, desData);
           // cout << res.num_success << "/" << res.num_total << endl;
           double loss = net.get_loss<mse>(inData, desData);
           cout << "Epoch = " << epo << endl;
           epo++;
           cout << "Loss = " << loss << endl << endl;
           // ofs << nn;
         }); // batch size 4, 1000 epochs

    cout << "!!!" << endl;
    int k = numOfFiles + 2;
    for (auto& tn : inData)
    {
        // auto res_label = net.predict_label(tn);
        cout << k << endl;
        auto res = net.predict(tn);
        cout << res[0] << endl;

        ofstream predictionFile(string("../prediction/") + string(argv[k]) + string("Pred"));

        for (int i = 0; i < matrixDim; i++)
            for (int j = 0; j < 4; j++)
                predictionFile << res[i * 4 + j] << " ";
            predictionFile << endl;

        k++;
        // std::cout << "In: (" << tn[0] << "," << tn[1] << ") Prediction: " << res_label << std::endl;

        predictionFile.close();
    }
    
    return 0;
}
