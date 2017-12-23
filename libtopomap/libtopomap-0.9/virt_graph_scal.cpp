/*
 * Copyright (c) 2008 The Trustees of Indiana University and Indiana
 *                    University Research and Technology
 *                    Corporation.  All rights reserved.
 *
 * Author(s): Torsten Hoefler 
 *
 */
#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <vector>
#include <utility>

#include "virt_graph_scal.hpp"

typedef struct commkey {
  std::vector<int> out, outw, in, inw;
  bool weighted;
} STI_Comminfo;

/* the keyval (global) */
static __thread int gkeyval=MPI_KEYVAL_INVALID;

static int STI_Key_copy(MPI_Comm oldcomm, int keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, int *flag) {
    /* delete the attribute in the new comm  - it will be created at the
     *    * first usage */
    *flag = 0;

      return MPI_SUCCESS;
}

static int STI_Key_delete(MPI_Comm comm, int keyval, void *attribute_val, void *extra_state) {
  STI_Comminfo *comminfo;

  if(keyval == gkeyval) {
    comminfo=(STI_Comminfo*)attribute_val;
    free((void*)comminfo);
  } else {
    printf("Got wrong keyval!(%i)\n", keyval);
  }

  return MPI_SUCCESS;
}


int MPIX_Dist_graph_create_adjacent(MPI_Comm comm_old, 
              int indegree, int sources[], int sourceweights[],
              int outdegree, int destinations[], int destweights[],
              MPI_Info info, int reorder, MPI_Comm *comm_dist_graph) {
  int res = MPI_Comm_dup(comm_old, comm_dist_graph);
  if((MPI_SUCCESS != res)) { printf("Error in MPI_Comm_dup(1) (%i)\n", res); return res; }

  if(MPI_KEYVAL_INVALID == gkeyval) {
    res = MPI_Keyval_create(STI_Key_copy, STI_Key_delete, &(gkeyval), NULL);
    if((MPI_SUCCESS != res)) { printf("Error in MPI_Keyval_create(1) (%i)\n", res); return res; }
  }

  STI_Comminfo *comminfo = new struct commkey(); 

  if(sourceweights == MPIX_UNWEIGHTED || destweights == MPIX_UNWEIGHTED) comminfo->weighted=false;
  else comminfo->weighted=true;
  
  for(int i=0; i<indegree; ++i) {
    comminfo->in.push_back(sources[i]);
    if(sourceweights != MPIX_UNWEIGHTED) comminfo->inw.push_back(sourceweights[i]);
  }
  for(int i=0; i<outdegree; ++i) {
    comminfo->out.push_back(destinations[i]);
    if(destweights != MPIX_UNWEIGHTED) comminfo->outw.push_back(destweights[i]);
  }

  /* put the new attribute to the comm */
  res = MPI_Attr_put(*comm_dist_graph, gkeyval, comminfo);
  if((MPI_SUCCESS != res)) { printf("Error in MPI_Attr_put() (%i)\n", res); return res; }

  return MPI_SUCCESS;
}

int MPIX_Dist_graph_create(MPI_Comm comm_old, int n, int nodes[], int degrees[], 
                          int targets[], int weights[], MPI_Info info,
                          int reorder, MPI_Comm *newcomm) {
  /* build arrays for flexible interface */
  int r, p;
  MPI_Comm_rank(comm_old, &r);
  MPI_Comm_size(comm_old, &p);

  int res = MPI_Comm_dup(comm_old, newcomm);
  if((MPI_SUCCESS != res)) { printf("Error in MPI_Keyval_create() (%i)\n", res); return res; }

  // htor: this is a silly AMPI workaround!!! 
  // begin original
  //if(MPI_KEYVAL_INVALID == gkeyval) {
  // end original
  // begin workaround
  {
  // end workaround
    res = MPI_Keyval_create(STI_Key_copy, STI_Key_delete, &(gkeyval), NULL);
    if((MPI_SUCCESS != res)) { printf("Error in MPI_Keyval_create() (%i)\n", res); return res; }
  }
  //printf("[%i] created keyval: %i\n", r, gkeyval);

  std::vector<std::vector<int> > rout(p), rin(p);
  int index=0;
  assert(n<=p);
  for(int i=0; i<n; ++i) {
    assert(nodes[i] < p);
    for(int j=0; j<degrees[i]; ++j) {
      assert(nodes[i] < p);
      rout[nodes[i]].push_back(targets[index]);
      if(weights != MPIX_UNWEIGHTED) rout[nodes[i]].push_back(weights[index]);

      assert(targets[index] < p);
      rin[targets[index]].push_back(nodes[i]);
      if(weights != MPIX_UNWEIGHTED) rin[targets[index]].push_back(weights[index]);
      
      index++;
    }
  }

  std::vector<int> redscat(2*p);
  std::vector<int> redscatres(2);
  std::vector<int> cnts(p);
  for(int i=0; i<p; ++i) {
    if(rin[i].size()) redscat[2*i]=1; else redscat[2*i]=0;
    if(rout[i].size()) redscat[2*i+1]=1; else redscat[2*i+1]=0;
    cnts[i]=2;
  }

  MPI_Reduce_scatter(&redscat[0], &redscatres[0], &cnts[0], MPI_INT, MPI_SUM, comm_old);

  std::vector<MPI_Request> reqs;
  for(int i=0; i<p; ++i) {
    if(rin[i].size()) {
      reqs.resize(reqs.size()+1);
      MPI_Isend(&rin[i][0], rin[i].size(), MPI_INT, i, 99, comm_old, &reqs[reqs.size()-1]);
      //for(int x=0; x<rin[i].size(); ++x) if(rin[i][x] > p) printf("before in %i\n", rin[i][x]);
      //printf("[%i] sending %i ints to %i\n", r, rin[i].size(), i);
    }
    if(rout[i].size()) {
      reqs.resize(reqs.size()+1);
      MPI_Isend(&rout[i][0], rout[i].size(), MPI_INT, i, 98, comm_old, &reqs[reqs.size()-1]);
    }
  }

  STI_Comminfo *comminfo = new struct commkey(); 

  if(weights == MPIX_UNWEIGHTED) comminfo->weighted=false;
  else comminfo->weighted=true;

  assert(redscatres[0]<=p);
  for(int i=0; i<redscatres[0]; ++i) {
    MPI_Status stat;
    /* receive incoming edges */
    MPI_Probe(MPI_ANY_SOURCE, 99, comm_old, &stat);
    int count;
    MPI_Get_count(&stat, MPI_INT, &count);
    assert(count > 0);

    std::vector<int> buf(count);
    MPI_Recv(&buf[0], count, MPI_INT, stat.MPI_SOURCE, 99, comm_old, MPI_STATUS_IGNORE);

    if(weights != MPIX_UNWEIGHTED) {
      for(int j=0; j<count/2; j++) {
        comminfo->in.push_back(buf[2*j]);
        comminfo->inw.push_back(buf[2*j+1]);
      }
    } else {
      for(int j=0; j<count; j++) {
        comminfo->in.push_back(buf[j]);
        //if(buf[j] > p) printf("[%i] in %i (from: %i) pos: %i of %i\n", r, buf[j], stat.MPI_SOURCE, j, count);
      }
    }
  }

  assert(redscatres[1]<=p);
  for(int i=0; i<redscatres[1]; ++i) {
    /* receive outgoung edges */
    MPI_Status stat;
    MPI_Probe(MPI_ANY_SOURCE, 98, comm_old, &stat);
    int count;
    MPI_Get_count(&stat, MPI_INT, &count);
    std::vector<int> buf(count);
    MPI_Recv(&buf[0], count, MPI_INT, stat.MPI_SOURCE, 98, comm_old, MPI_STATUS_IGNORE);
    if(weights != MPIX_UNWEIGHTED) {
      for(int j=0; j<count/2; j++) {
        comminfo->out.push_back(buf[2*j]);
        comminfo->outw.push_back(buf[2*j+1]);
      }
    } else {
      for(int j=0; j<count; j++) {
        comminfo->out.push_back(buf[j]);
        //if(buf[j] > p) printf("out %i\n", buf[j]);
      }
    }
  }

  // silly AMPI doesn't support MPI_STATUSES_IGNORE
  std::vector<MPI_Status> stats(reqs.size());
  MPI_Waitall(reqs.size(), &reqs[0], &stats[0]);

  /* put the new attribute to the comm */
  res = MPI_Attr_put(*newcomm, gkeyval, comminfo);
  if((MPI_SUCCESS != res)) { printf("[%i] Error in MPI_Attr_put(1) (%i)\n", r, res); return res; }

  redscatres[0] = 0;

  return MPI_SUCCESS;
}

int MPIX_Dist_graph_neighbors_count(MPI_Comm comm, int *inneighbors, int *outneighbors, int *weighted) {
  STI_Comminfo *comminfo; 

  int flag;
  int res = MPI_Attr_get(comm, gkeyval, &comminfo, &flag);
  if((MPI_SUCCESS != res)) { printf("Error in MPI_Attr_get(2) (%i)\n", res); return res; }

  *inneighbors = comminfo->in.size();
  *outneighbors = comminfo->out.size();
  
  if(comminfo->weighted) *weighted=1;
  else *weighted=0;

  return MPI_SUCCESS;
}

int MPIX_Dist_graph_neighbors(MPI_Comm comm, int maxindegree, int sources[], 
                       int sourceweights[], int maxoutdegree, 
                       int destinations[], int destweights[]) {
  
  int flag;
  STI_Comminfo *comminfo; 
  int res = MPI_Attr_get(comm, gkeyval, &comminfo, &flag);
  if((MPI_SUCCESS != res)) { printf("Error in MPI_Attr_get(3) (%i)\n",res); return res; }

  if(maxindegree < comminfo->in.size() || maxoutdegree < comminfo->out.size()) return MPI_ERR_COUNT;
  //printf("htor %i %i\n", comminfo->in.size(), comminfo->out.size());

  for(int i=0; i<comminfo->in.size(); ++i) {
    sources[i] = comminfo->in[i];
    if(comminfo->weighted) sourceweights[i] = comminfo->inw[i];
  }
  for(int i=0; i<comminfo->out.size(); ++i) {
    destinations[i] = comminfo->out[i];
    if(comminfo->weighted) destweights[i] = comminfo->outw[i];
  }
  
  return MPI_SUCCESS;
}

#ifdef TEST
int main(int argc, char *argv[]) {
  
  int rank, p;
  int *nodes, *degrees, *tgtnodes;
  int inneighbors, outneighbors;
  int i, n, e;
  int *inneigh, *outneigh, *weights;
  MPI_Comm newcomm;

  MPI_Init(&argc,&argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  n = (rank==p-1) ? 3 : 2;
  e = (rank==p-1) ? p+3 : 3; /* extra edges */
  nodes    = (int*)malloc(n*sizeof(int));
  degrees  = (int*)malloc(n*sizeof(int));
  tgtnodes = (int*)malloc(e*sizeof(int));

  nodes[0]    = rank;
  degrees[0]  = 2;
  tgtnodes[0] = (rank+1)%p; // a ring
  tgtnodes[1] = 0; // a star into 0 

  nodes[1]    = (rank+1)%p;
  degrees[1]  = 1;
  tgtnodes[2] = rank; // a double ring

  if (rank==p-1) { // star out of p-1
    nodes[2] = rank;
    degrees[2] = p;
    for (i=0; i<p; i++) tgtnodes[3+i] = i; 
  }

  weights = (int*)malloc(sizeof(int)*e);
  for(i=0;i<e;i++) weights[i] = 1;

  MPIX_Dist_graph_create(MPI_COMM_WORLD, n, nodes, degrees, tgtnodes, 
                           weights, MPI_INFO_NULL, 0, &newcomm);

  MPIX_Dist_neighbors_count(newcomm, &inneighbors, &outneighbors);
  printf("rank: %i has %i inneighbors and %i outneighbors\n", rank, inneighbors, outneighbors);

  inneigh =  (int*)malloc(inneighbors*sizeof(int));
  outneigh = (int*)malloc(outneighbors*sizeof(int));
  int *inneighw =  (int*)malloc(inneighbors*sizeof(int));
  int *outneighw = (int*)malloc(outneighbors*sizeof(int));
  MPIX_Dist_neighbors(newcomm, inneighbors, inneigh, inneighw, outneighbors, outneigh, outneighw);

  printf("rank %i has outneighbors:", rank);
  for (i=0; i<outneighbors; i++) printf(" %i", outneigh[i]);
  printf(" and inneighbors:");
  for (i=0; i<inneighbors; i++) printf(" %i", inneigh[i]);
  printf("\n");

  MPI_Finalize();
} 
#endif
