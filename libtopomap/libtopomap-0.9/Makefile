include Makefile.inc
#### See Makefile.inc for variables!
####
CFLAGS=-I$(PARMETIS) -DFAKE_MPI22 -I$(METIS)/Lib $(SCOTCH_INC)
LDFLAGS=libtopomap.a $(PARMETIS)/libparmetis.a $(PARMETIS)/libmetis.a  $(SCOTCH_LIB)
VERSION=0.9

all: test

virt_graph_scal.o: virt_graph_scal.cpp virt_graph_scal.hpp
	$(CXX) $(CFLAGS) -c -o $@ $< 

test.o: test.cpp
	$(CXX) $(CFLAGS) -c -o $@ $<

test: test.o virt_graph_scal.o libtopomap.a
	$(CXX) $(CFLAGS) -o $@ test.o virt_graph_scal.o $(LDFLAGS)

topoparser.o: topoparser.cpp libtopomap.hpp
	$(CXX) $(CFLAGS) -c -o $@ $<

libtopomap.o: libtopomap.cpp libtopomap.hpp
	$(CXX) $(CFLAGS) -c -o $@ $<

bgptopo.o: bgptopo.cpp libtopomap.hpp
	$(CXX) $(CFLAGS) -c -o $@ $<

rcm.o: rcm.c
	$(CXX) $(CFLAGS) -c -o $@ $<

libtopomap.a: topoparser.o libtopomap.o rcm.o bgptopo.o
	ar r $@ topoparser.o libtopomap.o rcm.o bgptopo.o
	ranlib $@

clean:
	rm -f virt_graph_scal.o topoparser.o libtopomap.o libtopomap.a test test.o rcm.o bgptopo.o

pack:
	rm -rf libtopomap-$(VERSION)
	mkdir -p libtopomap-$(VERSION)/matvec-test
	cp README LICENSE 3x3x3.fake 3x3x3.map bgptopo.cpp libtopomap.cpp \
		libtopomap.hpp Makefile Makefile.inc MersenneTwister.h rcm.c \
		rcm.h test.cpp topoparser.cpp virt_graph_scal.cpp virt_graph_scal.hpp \
		libtopomap-$(VERSION)
	cp matvec-test/Makefile matvec-test/reader.cc libtopomap-$(VERSION)/matvec-test
	tar czf libtopomap-$(VERSION).tgz libtopomap-$(VERSION)
