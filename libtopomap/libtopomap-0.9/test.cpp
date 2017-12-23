#include "libtopomap.hpp"
#include <math.h>

void rank2xyz(int maxx, int maxy, int maxz, int rank, int *x, int *y, int *z) {
  *x = rank / (maxy*maxz);
  rank = rank - *x*maxy*maxz;
  *y = rank / maxz;
  rank = rank - *y*maxz;
  *z = rank;
}
void xyz2rank(int maxx, int maxy, int maxz, int x, int y, int z, int *rank) {
  *rank = x*maxy*maxz + y*maxz + z;
}


int main(int argc, char **argv) {

  MPI_Init(NULL, NULL);
  int r,p;
  MPI_Comm comm=MPI_COMM_WORLD;
  MPI_Comm_size(comm, &p);
  MPI_Comm_rank(comm, &r);

  if (argc <= 1) {
    fprintf(stderr, "%s <expr>\n", argv[0]);
    return 1;
  }
  if (argc == 3) {
    TPM_Fake_names_file = argv[2];
    if(!p) printf("using fake topology file %s\n", TPM_Fake_names_file);
  }

  MPI_Comm distgr;

#define CUBE2D
#ifdef RING
  /* CREATE DUMMY DIST_GRAPH COMMUNICATOR 
   * will consist of vertices that are layed out in circles of a
   * specific size (scrambled) */
  int circsize=12, offset=1; 
  std::vector<int> sources(1);
  sources[0]=(r+1)%p;
  std::vector<int> degrees(1);
  degrees[0]=2;
  std::vector<int> destinations(2);
  destinations[0] = (((r+1)%circsize)%p+circsize*(r/circsize)+1)%p;
  destinations[1] = (((r-1+circsize)%circsize)%p+circsize*(r/circsize)+1)%p;

  MPIX_Dist_graph_create(comm, 1, &sources[0], &degrees[0], &destinations[0], (int*)MPIX_UNWEIGHTED, 
                        MPI_INFO_NULL, 0, &distgr);
  /* END CREATE DUMMY DIST_GRAPH COMMUNICATOR */
#endif

#ifdef CUBE2D
  int size=(int)floor(sqrt(p)); 
  if(!r) printf("cube size: %ix%i\n", size, size);

  std::vector<int> sources(1);
  sources[0]=r;
  std::vector<int> degrees(1);
  degrees[0]=4;
  std::vector<int> destinations(4);

  /* find my x,y pair */
  int rank=0, myx, myy;
  std::vector<std::vector<int> > pos2rank(size);
  for(int i=0; i<size; ++i) pos2rank[i].resize(size);

  for(int x=0; x<size; ++x) {
    for(int y=0; y<size; ++y) {
      if(rank == r) {
        myx=x;
        myy=y;
      }
      pos2rank[x][y] = rank++;
    }
  }

  if(r < size*size) {
//#define MESH
#ifdef MESH
    int peerx[2], peery[2];
    peerx[0] = myx-1;
    peerx[1] = myx+1;
    if(peerx[1] == size) peerx[1] = -1;
    peery[0] = myy-1;
    peery[1] = myy+1;
    if(peery[1] == size) peery[1] = -1;

    int neighbor = 0;

    if(peerx[0] != -1) destinations[neighbor++] = pos2rank[peerx[0]][myy];
    if(peerx[1] != -1) destinations[neighbor++] = pos2rank[peerx[1]][myy];
    if(peery[0] != -1) destinations[neighbor++] = pos2rank[myx][peery[0]];
    if(peery[0] != -1) destinations[neighbor++] = pos2rank[myx][peery[1]];
    degrees[0] = neighbor;
#else // torus
    destinations[0] = pos2rank[(myx+size-1)%size][myy];
    destinations[1] = pos2rank[(myx+1)%size][myy];
    destinations[2] = pos2rank[myx][(myy+size-1)%size];
    destinations[3] = pos2rank[myx][(myy+1)%size];
#endif
  } else {
    degrees[0]=0;
  }

  MPIX_Dist_graph_create(comm, 1, &sources[0], &degrees[0], &destinations[0], (int*)MPIX_UNWEIGHTED, 
                        MPI_INFO_NULL, 0, &distgr);
#endif

#ifdef CUBE3D
  const int x=3,y=3,z=3;
  int tgt;

  std::vector<int> sources(1);
  sources[0] = r;
  std::vector<int> degrees(1);
  degrees[0] = 6;
  std::vector<int> destinations(6);
  int mx, my, mz;
  rank2xyz(x,y,z,r,&mx,&my,&mz);
  printf("[%i] %i %i %i\n", r, mx, my, mz);
  
  xyz2rank(x,y,z,(mx+x+1)%x,my,mz,&tgt);
  destinations[0] = tgt;
  xyz2rank(x,y,z,(mx+x-1)%x,my,mz,&tgt);
  destinations[1] = tgt;
  xyz2rank(x,y,z,mx,(my+y+1)%y,mz,&tgt);
  destinations[2] = tgt;
  xyz2rank(x,y,z,mx,(my+y-1)%y,mz,&tgt);
  destinations[3] = tgt;
  xyz2rank(x,y,z,mx,my,(mz+z+1)%z,&tgt);
  destinations[4] = tgt;
  xyz2rank(x,y,z,mx,my,(mz+z-1)%z,&tgt);
  destinations[5] = tgt;

  
  
  MPIX_Dist_graph_create(comm, 1, &sources[0], &degrees[0], &destinations[0], (int*)MPIX_UNWEIGHTED, 
                        MPI_INFO_NULL, 0, &distgr);

#endif

  TPM_Write_graph_comm(distgr, "graphcomm.dot");

  int myrank;
  //int ret = TPM_Topomap(distgr, argv[1], 0, &myrank);
  int ret = TPM_Topomap(distgr, (const char*)argv[1], 0, &myrank);
  if(ret != 0) {
    if(!r) printf("topomap failed -- aborting\n");
    MPI_Finalize();
    return 0;
  }

  const int len=1024;
  char name[len];
  gethostname(name, len);
  //printf("%s (%i -> %i)\n", name, r, myrank);
  
  int indegree, outdegree, weighted;
  MPIX_Dist_graph_neighbors_count(distgr, &indegree, &outdegree, &weighted);
  std::vector<int> in(indegree), out(outdegree), inw(indegree), outw(outdegree);
  MPIX_Dist_graph_neighbors(distgr, indegree, &in[0], &inw[0], outdegree, &out[0], &outw[0]);

  /* create reordered comm */
  MPI_Comm reorderedcomm;
  MPI_Comm_split(comm, 0, myrank, &reorderedcomm);

  /* do benchmark - send along edges */
  double t[2];

  static const int bsize=1024*10240;
  static const int trials=5;
  std::vector<void*> sbufs(outdegree), rbufs(indegree);
  for(int i=0; i<outdegree; ++i) sbufs[i] = malloc(bsize);
  for(int i=0; i<indegree; ++i) rbufs[i] = malloc(bsize);
  std::vector<MPI_Request> sreqs(outdegree), rreqs(indegree);
  
  MPI_Barrier(comm);

  t[0] = -MPI_Wtime();
  for(int i=0; i < trials; ++i) {
    for(int j=0; j<in.size(); ++j) {
      MPI_Irecv(rbufs[j], bsize, MPI_BYTE, in[j], 0, distgr, &rreqs[j]);
    }
    for(int j=0; j<out.size(); ++j) {
      MPI_Isend(sbufs[j], bsize, MPI_BYTE, out[j], 0, distgr, &sreqs[j]);
    }
    // silly AMPI doesn't support MPI_STATUSES_IGNORE
    std::vector<MPI_Status> rstats(rreqs.size());
    std::vector<MPI_Status> sstats(sreqs.size());
    MPI_Waitall(rreqs.size(), &rreqs[0], &rstats[0]);
    MPI_Waitall(sreqs.size(), &sreqs[0], &sstats[0]);
  }
  t[0] += MPI_Wtime();

  if(!r) printf("finished stage one, rearranging edge connectivity!\n");

  /* now we need to send our neighbor list to the host that has our rank
   * now -- first figure out the permutation in a non-scalable way :) */
  std::vector<int> permutation(p);
  MPI_Allgather(&myrank, 1, MPI_INT, &permutation[0], 1, MPI_INT, comm);
  int rpeer=permutation[r]; // new rank r in reorderedcomm!
  int speer=0; while(permutation[speer++] != r); speer--;
  //if(!r) { printf("permutation: "); for(int i=0; i<p; ++i) printf("%i ", permutation[i]); printf("\n"); }
  //printf("[%i] >%i <%i\n", r, speer, rpeer);
  MPI_Request reqs[2];
  MPI_Isend(&in[0], in.size(), MPI_INT, speer, 1, comm, &reqs[0]);
  //printf("%i sending in %i %i to %i\n", r, in[0], in[1], speer);
  MPI_Isend(&out[0], out.size(), MPI_INT, speer, 2, comm, &reqs[1]);
  //printf("%i sending out %i %i to %i\n", r, out[0], out[1], speer);
  MPI_Status stat;
  /* tag == 1 -> in edges */
  MPI_Probe(rpeer, 1, comm, &stat);
  int count;
  MPI_Get_count(&stat, MPI_INT, &count);
  std::vector<int> rin(count);
  MPI_Recv(&rin[0], count, MPI_INT, rpeer, 1, comm, MPI_STATUS_IGNORE);
  //printf("%i recvd in %i %i from %i\n", r, rin[0], rin[1], rpeer);
  /* tag == 2 -> out edges */
  MPI_Probe(rpeer, 2, comm, &stat);
  MPI_Get_count(&stat, MPI_INT, &count);
  std::vector<int> rout(count);
  MPI_Recv(&rout[0], count, MPI_INT, rpeer, 2, comm, MPI_STATUS_IGNORE);
  //printf("%i recvd out %i %i from %i\n", r, rout[0], rout[1], rpeer);
 
  // silly AMPI doesn't support MPI_STATUSES_IGNORE
  std::vector<MPI_Status> stats(2);
  MPI_Waitall(2, reqs, &stats[0]);

  int newr;
  MPI_Comm_rank(reorderedcomm, &newr);

  /* reset all data structures for new permutation */ 
  sreqs.resize(rout.size());
  rreqs.resize(rin.size());
  for(int i=0; i<outdegree; ++i) free(sbufs[i]);
  for(int i=0; i<indegree; ++i) free(rbufs[i]);
  sbufs.resize(rout.size());
  rbufs.resize(rin.size());
  for(int i=0; i<sbufs.size(); ++i) sbufs[i] = malloc(bsize);
  for(int i=0; i<rbufs.size(); ++i) rbufs[i] = malloc(bsize);

  //printf("%i starting reordered comm %i %i (%i %i)\n", r, rin.size(), rout.size(), permutation[r], newr);
  MPI_Barrier(comm);

  t[1] = -MPI_Wtime();
  for(int i=0; i < trials; ++i) {
    for(int j=0; j<rin.size(); ++j) {
      MPI_Irecv(rbufs[j], bsize, MPI_BYTE, rin[j], 0, reorderedcomm, &rreqs[j]);
    }
    for(int j=0; j<rout.size(); ++j) {
      MPI_Isend(sbufs[j], bsize, MPI_BYTE, rout[j], 0, reorderedcomm, &sreqs[j]);
    }
    // silly AMPI doesn't support MPI_STATUSES_IGNORE
    std::vector<MPI_Status> rstats(rreqs.size());
    std::vector<MPI_Status> sstats(sreqs.size());
    MPI_Waitall(rreqs.size(), &rreqs[0], &rstats[0]);
    MPI_Waitall(sreqs.size(), &sreqs[0], &sstats[0]);
  }
  t[1] += MPI_Wtime();

  double rt[2];
  MPI_Reduce(t, rt, 2, MPI_DOUBLE, MPI_MAX, 0, comm);
  if(!r) printf("%lf %lf \n", rt[0], rt[1]);

  MPI_Finalize();
}
