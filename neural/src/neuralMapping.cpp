#include "tiny_dnn/tiny_dnn.h"
#include <fstream>
#include <cmath>

using namespace std;
using namespace tiny_dnn;
using namespace tiny_dnn::activation;

#define EPS 0.0001


int main(int argc, char** argv)
{
    /*<numOfFiles> <dim> <matrixFiles> <mappingFiles>*/
    int numOfFiles = atoi(argv[1]);
    long unsigned int matrixDim = atoi(argv[2]);
    int lenMapStr = 4;

    std::vector<vec_t> inData;
    for (int k = 3; k < numOfFiles + 3; k++)
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

    std::vector<vec_t> desData;
    for (int k = numOfFiles + 3; k < argc; k++)
    {
        ifstream mappingFile(argv[k]);
        vec_t mappingVec;

        double tmp = 0;
        for (int i = 0; i < matrixDim; i++)
        {
            for (int j = 0; j < lenMapStr; j++)
            {
                mappingFile >> tmp;
                if (tmp != 0)
                    mappingVec.push_back(1 / tmp);
                else
                    mappingVec.push_back(tmp);
            }

            // mappingFile >> tmp;
        }

        desData.push_back(mappingVec);
        mappingVec.clear();

        mappingFile.close();
    }

    int max = 0;
    if (matrixDim <= 8)
        max = 2;
    else
    if ((matrixDim <= 16) || (matrixDim <= 32) || (matrixDim <= 64))
        max = 4;
    else
    if ((matrixDim <= 128) || (matrixDim <= 256) || (matrixDim <= 512))
        max = 8;
    else
    if ((matrixDim <= 1024) || (matrixDim <= 2048))
        max = 16;

    // network<sequential> net = make_mlp<sigmoid>({matrixDim * matrixDim, matrixDim * matrixDim, matrixDim * 4});
    network<sequential> net;
    net << fully_connected_layer(matrixDim * matrixDim, matrixDim * 8, true) << sigmoid()
        // << fully_connected_layer(matrixDim * 2, matrixDim * 2, true) << sigmoid()
        // << fully_connected_layer(matrixDim * 2, matrixDim * 2, true) << sigmoid()
        << fully_connected_layer(matrixDim * 8, matrixDim * lenMapStr, true) << sigmoid();


    int epo = 0;
    timer t;
    adagrad optimizer; // use gradient_descent?
    net.fit<mse>(optimizer, inData, desData, 5, 500,
    // called for each mini-batch
         [&](){
           cout << t.elapsed() << endl;
           t.restart();
         },
         // called for each epoch
         [&](){
           double loss = net.get_loss<mse>(inData, desData);
           cout << "Epoch = " << epo << endl;
           epo++;
           cout << "Loss = " << loss << endl << endl;
         });

    int k = numOfFiles + 3;
    for (auto& tn : inData)
    {
        cout << k << endl;
        auto res = net.predict(tn);

        ofstream predictionFile(string("./prediction/") + string(argv[k]).substr(10, string(argv[k]).size() - 10) + string("Pred"));

        for (int i = 0; i < matrixDim; i++)
        {
            for (int j = 0; j < lenMapStr; j++)
            {
                cout << res[i * lenMapStr + j] << " ";
                if (fabs(res[i * lenMapStr + j] - 0) > EPS)
                {
                    double tmp = 1 / res[i * lenMapStr + j];
                    // if (fabs(tmp - (max - 1)) < EPS)
                    if (roundf(tmp) > (max - 1))
                        predictionFile << 0 << " ";
                    else
                        predictionFile << roundf(tmp) << " ";
                }
                else
                    predictionFile << 0 << " ";
            }
            // predictionFile << 0 << " " << endl;
            cout << endl;
            predictionFile << endl;
        }

        k++;

        predictionFile.close();
    }
    
    return 0;
}
