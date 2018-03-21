#include "tiny_dnn/tiny_dnn.h"
#include <fstream>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace tiny_dnn;
using namespace tiny_dnn::activation;

#define EPS 0.0001


// bool pairsort (pair <int, int> f, pair <int, int> s)
// {
//     return (f.second < s.second);
// }


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
        std::vector<int> tmpVec;
        for (int i = 0; i < matrixDim; i++)
        {
            for (int j = 0; j < matrixDim; j++)
            {
                matrixFile >> tmp;

                // matrixVec.push_back(tmp);
                tmpVec.push_back(tmp);
            }

            std::vector<pair <int, int> > pairVec;
            pair <int, int> tmpPair;
            for (int k = 0; k < matrixDim; k++)
            {
                tmpPair = make_pair(k, tmpVec[k]);
                pairVec.push_back(tmpPair);
            }
            sort(pairVec.begin(), pairVec.end(), [](pair <int, int> f, pair <int, int> s) {return (f.second < s.second);});

            for (int k = 0; k < matrixDim; k++)
            {
                if (pairVec[k].second == 0)
                    tmpVec[pairVec[k].first] = 0;
                else
                    tmpVec[pairVec[k].first] = k + 1;
            }

            for (int k = 0; k < matrixDim; k++)
                matrixVec.push_back(tmpVec[k]);
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


    int numFilt = 3;
    network<sequential> net;

    net << convolutional_layer(matrixDim, matrixDim, 4, 1, numFilt, padding::same)
        << max_pooling_layer(matrixDim, matrixDim, numFilt, 2)
        << convolutional_layer(matrixDim / 2, matrixDim / 2, 4, numFilt, numFilt, padding::same)
        << max_pooling_layer(matrixDim / 2, matrixDim / 2, numFilt, 2)
        << convolutional_layer(matrixDim / 4, matrixDim / 4, 4, numFilt, numFilt, padding::same)
        << max_pooling_layer(matrixDim / 4, matrixDim / 4, numFilt, 2)
        // << dropout_layer(matrixDim * matrixDim * numFilt / 64, 0.3)
        << fully_connected_layer(matrixDim * matrixDim * numFilt / 64, matrixDim * matrixDim * numFilt / 128) << sigmoid()
        << fully_connected_layer(matrixDim * matrixDim * numFilt / 128, matrixDim * lenMapStr) << sigmoid();


    int epo = 0;
    timer t;

    // adagrad optimizer;
    adam optimizer;
    // adamax optimizer;

    // net.fit<cross_entropy>(optimizer, inData, desData, 3, 500,

    net.fit<mse>(optimizer, inData, desData, 3, 1000,
    // called for each mini-batch
         [&](){
           // cout << t.elapsed() << endl;
           // t.restart();
         },
         // called for each epoch
         [&](){
           // double loss = net.get_loss<cross_entropy>(inData, desData);

           double loss = net.get_loss<mse>(inData, desData);
           cout << "Epoch = " << epo++ << endl;
           cout << "Loss = " << loss << endl << endl;
         });

    net.save("nets/net1");

    // net.save("nets/net2");

    int k = numOfFiles + 3;
    for (auto& tn : inData)
    {
        // cout << k << endl;
        auto res = net.predict(tn);

        int namePos = string(argv[k]).find_last_of("/\\");
        ofstream predictionFile(string("./prediction/") + string(argv[k]).substr(namePos + 1) + string("Pred"));

        for (int i = 0; i < matrixDim; i++)
        {
            for (int j = 0; j < lenMapStr; j++)
            {
                // cout << res[i * lenMapStr + j] << " ";
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
            // cout << endl;
            predictionFile << endl;
        }

        k++;

        predictionFile.close();
    }
    
    return 0;
}
