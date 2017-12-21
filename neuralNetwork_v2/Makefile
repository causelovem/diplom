all:
	g++ -DLOG -DNO_MPI -std=c++11 -fopenmp -I${PWD}/tiny-dnn main.cpp -O3 -o NeuralNetwork
tree:
	g++ -M -DCNN_NO_SERIALIZATION -DDEBUG -std=c++11 -fopenmp -I${PWD}/tiny-dnn main.cpp -O3 -o NeuralNetwork | grep '#' | cut -d' ' -f3 | sort | uniq
parallel:
	mpicxx -DLOG -std=c++11 -fopenmp -I${PWD}/tiny-dnn main.cpp -O3 -o NeuralNetwork
lomonosov:
	mpicxx -w -DLOMONOSOV -std=c++11 -openmp -I${PWD}/tiny-dnn main.cpp -O3 -o NeuralNetwork
modules:
	module add intel/15.0.090 impi/5.0.1 slurm/2.5.6
clean:
	rm -rf NeuralNetwork
