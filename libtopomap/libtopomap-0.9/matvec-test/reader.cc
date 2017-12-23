#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <parmetis.h>

#include <vector>
#include <algorithm>
#include <map>

#include "libtopomap.hpp"



bool smaller(std::pair<int,int> elem1, std::pair<int,int> elem2) {
     return elem1.first < elem2.first;
}

typedef std::pair<int,int> twoint;

int main(int argc, char** argv) {
  int m,n,nnz,start,end;
  int row, col;
  std::vector<std::pair<int,int> > mat;

  MPI_Init(NULL,NULL);

  int wr,ws;
  MPI_Comm_size(MPI_COMM_WORLD, &ws);
  MPI_Comm_rank(MPI_COMM_WORLD, &wr);

  if (argc <= 3) {
    fprintf(stderr, "%s <np (0 for all)> <matrix> <topomap file> [fake file]\n", argv[0]);
    return 1;
  }
  if (argc == 5) {
    TPM_Fake_names_file = argv[4];
    if(!wr) printf("using fake topology file %s\n", TPM_Fake_names_file);
  }

  int border,color,key=0;
  if(atoi(argv[1]) == 0) border=ws; else border = atoi(argv[1]);
  if(wr < border) color = 0; else color = 1;
  MPI_Comm comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, key, &comm);

  int p,r;
  MPI_Comm_size(comm, &p);
  MPI_Comm_rank(comm, &r);

  if(color == 0) {
    if(!r) printf("starting with %i processes (built: %s %s)\n", p, __DATE__, __TIME__);

    { 
      FILE *fp = fopen(argv[2], "r");
  
      const int LONGEST_LINE = 1024;
      char *buf = (char*)malloc(LONGEST_LINE+1);
      
      int firstline = 1;
      const int percinc=10;
      int lines = 0, perc=percinc;
      std::map<twoint,int,std::less<twoint> > edges;
      while(fgets(buf, LONGEST_LINE, fp) != NULL) {
    	
        if(buf[0] == '%') continue; // comment
        if(firstline) { // the very first line has a special meaning
          sscanf(buf,"%d %d %d\n",&m,&n,&nnz);
          assert(m==n);
          start = r*(n/p);
          end = (r+1)*(n/p);
          if(r == p-1) end = n;
          firstline = 0;
          if(!r) printf("reading matrix: %ix%i (nnz: %i)\n", m,n,nnz);
          continue;
        }
        sscanf(buf,"%d %d\n",&row,&col);
        // MM format starts with 1 instead of 0 -- correct!
        if(!(row>0 && row<=n)) printf("[%i] row: %i, n: %i\n", r, row,n);
        assert(row>0 && row<=n); assert(col>0 && col<=m);
        row--; col--;
        if(row != col) { // ParMeTiS doesn't like self-loops :-/
          if(row >= start && row < end) {
            // only insert if not already in graph
            std::pair<int,int> edge = std::make_pair(row,col);
            if(edges.find(edge) == edges.end()) {
              mat.push_back(std::make_pair(row, col));
              edges.insert(std::make_pair(edge,1));
            }
          }
          // read as symmetric graph!
          if(col >= start && col < end) {
            // only insert if not already in graph
            std::pair<int,int> edge = std::make_pair(col,row);
            if(edges.find(edge) == edges.end()) {
              mat.push_back(std::make_pair(col, row));
              edges.insert(std::make_pair(edge,1));
            }
          }
        }
        if(!r) if(++lines >= nnz*(double)perc/100) { printf("# read %i%% (%i) of lines\n", perc, lines); perc+=percinc; }
      }
      fclose(fp);
    }
  
    // matvec: z = v * A with z[i] = \sum_{j=0}^N (A[i,j] * v[j])
  
    //assert(mat.size() == nnz);
    int perproc = end-start;
    if(r<10) printf("[%i] finished reading matrix: %ix%i (nnz: %i), range: %i-%i (%i), elems: %i, \n", r,m,n,nnz,start,end,perproc,(int)mat.size());
  
    std::vector<idxtype>  vtxdist(p+1), // vertex distribution
                          xadj(perproc+1), // CSR index 
                          adjncy, // CSR list
                          part(perproc); // new index for each vertex
  
    // make sure that the initializer is bigger than all edge weights
    // (needed to guarantee that parmetis balances strictly!!!!
    idxtype initializer=1;
    std::vector<idxtype> vwgt(perproc,initializer); // vertex weights 
    std::vector<idxtype> adjwgt(mat.size(),initializer); // edge weights
  
    //for(int i=0; i<adjncy.size(); i++) printf("[%i] -> %i (%i)\n", r, adjncy[i], adjwgt[i]);
    int wgtflag=2; // 0 No weights, 1 Weights on the edges only, 2 Weights on the vertices only, 3 Weights on both the vertices and edges. 
    int numflag=0; // numbering scheme - 0 C-style numbering that starts from 0
    int ncon=1; // number of weights that each vertex has.
    int nparts = p; // number of partitions
    std::vector<float> tpwgts(ncon*nparts,1.0/nparts); // specify the fraction of vertex weight 
    std::vector<float> ubvec(ncon,1.05); // specify the imbalance tolerance for each vertex weight
    int options[3] = {0,0,0};
    int edgecut;
    // fill vtxdist 
    for(int i=0; i<p; ++i) {
      vtxdist[i] = i*(n/p);
    }
    vtxdist[p] = n;
  
    // fill xadj
    std::sort(mat.begin(), mat.end(), smaller);
  
    int idx=0;
    int num=0;
    for(int i=start; i<end; ++i) {
      while(idx < mat.size() && i == mat[idx].first) {
        num++;
        adjncy.push_back(mat[idx].second);
        idx++;
      }
      xadj[i-start+1] = num;
    }
    //printf("[%i] %i == %i? (start: %i, end: %i)\n", r, adjncy.size(), mat.size(), start, end);
    assert(adjncy.size() == mat.size());
   
    const int printrank=-1;
    const int printsize=100;
    if(r==printrank) for(int i=0; i<vtxdist.size(); ++i) printf("vtxdist[%i] = %i\n", i, vtxdist[i]);
    if(r==printrank) for(int i=0; i<std::min((int)mat.size(),printsize); ++i) printf("mat[%i] = %i,%i\n", i, mat[i].first, mat[i].second);
    if(r==printrank) for(int i=0; i<std::min((int)xadj.size(),printsize); ++i) printf("xadj[%i] = %i\n", i, xadj[i]);
    if(r==printrank) for(int i=0; i<std::min((int)adjncy.size(),printsize); ++i) printf("adjncy[%i] = %i\n", i, adjncy[i]);
  
    MPI_Barrier(comm);
    if(!r) printf("calling ParMETIS\n");
    ParMETIS_V3_PartKway(&vtxdist[0], &xadj[0], &adjncy[0], &vwgt[0],  &adjwgt[0], &wgtflag, &numflag, &ncon, 
        &nparts, &tpwgts[0], &ubvec[0], &options[0], &edgecut, &part[0], &comm);
    if(!r) printf("ParMETIS done, edgecut: %i\n", edgecut);
  
    if(r==printrank) for(int i=0; i<std::min((int)part.size(),printsize); ++i) printf("part[%i] = %i\n", i, part[i]);
  
    // the vector part has now the new position for each row of the matrix
    // and element of the two vectors (it can be used to permute all rows)
  
    // we now collect the whole permutation to all processes (this is of
    // course not scalable)
    std::vector<int> perm(n);
    std::vector<int> recvcounts(p), displs(p);
    for(int i=0; i<p; ++i) {
      recvcounts[i] = n/p;
      displs[i] = i*(n/p);
    }
    recvcounts[p-1] = n-(p-1)*(n/p);
  
    MPI_Allgatherv(&part[0], perproc, MPI_INT, &perm[0], &recvcounts[0], &displs[0], MPI_INT, comm);
  
    // now we build a p*p matrix with the communication volume for each
    // element we have (after applying the permutation)
    // THIS IS OF COURSE TOTALLY NOT SCALABLE!
    std::vector<int> volumes(p*p,0), rvolumes(p*p,0);
    for(int i=0; i<mat.size(); ++i) {
      int src = perm[mat[i].first];
      int dst = perm[mat[i].second];
      volumes[src+dst*p]++;
    }
    // now allreduce it (I feel really bad)
    MPI_Allreduce(&volumes[0], &rvolumes[0], p*p, MPI_INT, MPI_SUM, comm);
  
    // build distributed graph communicator with the right weights!
    std::vector<int> sources(1,r),
                     degrees(1,0),
                     destinations,
                     weights;
  
    for(int i=0; i<p; ++i) {
      if(rvolumes[r+i*p] != 0) {
        if(r == 0) printf("[%i] sending %i doubles to %i\n", r, rvolumes[r+i*p], i);
        if(r == i) continue; // skip self-edge
        degrees[0]++;
        destinations.push_back(i);
        weights.push_back(rvolumes[r+i*p]);
      }
    }
  
//if(!r) for(int i=0; i<destinations.size(); ++i) printf("%i %i\n", destinations[i], weights[i]);
  
    MPI_Comm distgr;
    MPIX_Dist_graph_create(comm, 1, &sources[0], &degrees[0], &destinations[0], &weights[0], MPI_INFO_NULL, 1, &distgr);
    TPM_Write_graph_comm(distgr, "./ltg.dot");
  
    for(int i=0; i<4; ++i) {
      if(i==0) setenv("TPM_STRATEGY", "greedy", 1);
      if(i==1) setenv("TPM_STRATEGY", "recursive", 1);
      if(i==2) setenv("TPM_STRATEGY", "rcm", 1);
      if(i==3) setenv("TPM_STRATEGY", "scotch", 1);

      int myrank;
      int ret = TPM_Topomap(distgr, argv[3], 0, &myrank);
    }

  //  MPI_Barrier(distgr); // this avoids IB RETRY_EXCEED errors on Odin (WTF!)

    //double before, after;
    //TPM_Benchmark_graphtopo(distgr, myrank, 800, &before, &after);
    //if(!r) printf("before: %f, after: %f\n", before, after);
  } 

  MPI_Finalize();
}
