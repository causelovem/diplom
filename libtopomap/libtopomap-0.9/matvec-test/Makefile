CXX=mpicxx
#LDFLAGS=../scotch_5.2.0beta7/src/libscotch/libscotch.a ../scotch_5.2.0beta7/src/libscotch/libscotcherr.a
LDFLAGS=/usr/lib/libscotch.a /usr/lib/libscotcherr.a
reader: reader.cc ../libtopomap.a
	make -C ..
	$(CXX) -g -DFAKE_MPI22 reader.cc -o reader -I.. -I../ParMetis-3.1.1/ ../ParMetis-3.1.1/libparmetis.a ../ParMetis-3.1.1/libmetis.a ../virt_graph_scal.o ../libtopomap.a $(LDFLAGS)

clean:
	rm -f reader
