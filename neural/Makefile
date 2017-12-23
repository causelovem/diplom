GREEDY_FILE = src/greedy
COM_MATRIX_GEN = src/com_matrix_gen
CHECK_PROG = src/prog_test
BUILD_MATRIX = src/build_matrix
BUILD_RAND_SEND = src/build_rand_send
NEURAL_MAPPING = src/neuralMapping
CXX = g++
CXXFLAGS = -g -O0 -Wall

all: $(COM_MATRIX_GEN) $(GREEDY_FILE) $(BUILD_MATRIX) $(BUILD_RAND_SEND) $(CHECK_PROG)
 
neural: $(NEURAL_MAPPING)

map:
	./bin/greedy matrix512 512

gen:
	./bin/com_matrix_gen matrix512 512

check:

	# for (( count = 0; count < 32; count++ )) do \
	# 	echo >> proc_$count ; \
	# done \

	# for (( count = 0; count < 256; count++ )) do echo >> proc_$count ; done
	
	rm -rf trace_folder/proc_*
	./bin/build_rand_send mas 256
	#mpirun -np 256 ./bin/prog_test '/home/causelove/mishenka/prog/kursach/greedy/trace_folder' mas
	mpisubmit.bg -np 256 -w 00:05:00 ./bin/prog_test -- '/gpfs/data/edu-cmc-sqi16y3-004/kursach/trace_folder' mas
	sleep 60
	./bin/build_matrix matrix 256 trace_folder/*
	./bin/greedy matrix 256 mmm

$(NEURAL_MAPPING): $(NEURAL_MAPPING).cpp
	g++ -O0 -DLOG -DNO_MPI -std=c++14 -fopenmp -I${PWD}/tiny-dnn src/neuralMapping.cpp -o bin/neuralMapping

$(GREEDY_FILE): $(GREEDY_FILE).cpp
	$(CXX) $(CXXFLAGS) $(GREEDY_FILE).cpp -o bin/greedy

$(COM_MATRIX_GEN): $(COM_MATRIX_GEN).cpp
	$(CXX) $(CXXFLAGS) $(COM_MATRIX_GEN).cpp -o bin/com_matrix_gen

$(BUILD_MATRIX): $(BUILD_MATRIX).cpp
	$(CXX) $(CXXFLAGS) $(BUILD_MATRIX).cpp -o bin/build_matrix

$(BUILD_RAND_SEND): $(BUILD_RAND_SEND).cpp
	$(CXX) $(CXXFLAGS) $(BUILD_RAND_SEND).cpp -o bin/build_rand_send

$(CHECK_PROG): $(CHECK_PROG).cpp
	mpicxx $(CXXFLAGS) $(CHECK_PROG).cpp -o bin/prog_test

clean:
	rm -rf bin/*
	rm -rf trace_folder/*
	rm -rf *.o