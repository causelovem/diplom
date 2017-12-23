#PARMETIS=MPIParMETIS/ParMetis-3.1.1
#METIS=MPIParMETIS/metis-4.0
#SCOTCH_INC=-Iscotch_5.2.0beta7/src/libscotch/ -DHAVE_SCOTCH
#SCOTCH_LIB=scotch_5.2.0beta7/src/libscotch/libscotch.a scotch_5.2.0beta7/src/libscotch/libscotcherr.a

PARMETIS=/home/causelove/diplom/libtopomap-0.9/ParMetis-3.1.1
METIS=/home/causelove/diplom/libtopomap-0.9/metis-4.0.3
SCOTCH_INC=-I/usr/include/scotch-int32 -DHAVE_SCOTCH
SCOTCH_LIB=/usr/lib/libscotch.a /usr/lib/libscotcherr.a

## MPICH2 / OpenMPI
CXX=mpicxx -DMPICH_IGNORE_CXX_SEEK -g -O0

## POE POWER 5
#CXX=mpCC -DMPICH_IGNORE_CXX_SEEK -g -O3

