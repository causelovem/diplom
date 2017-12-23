#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>
#include <vector>
#include <string>
#include <queue>
#include <numeric>
#include <algorithm>
#include <map>
#include <limits>
#include <list>
#include "virt_graph_scal.hpp"

#ifndef LIBTOPOMAP_HPP
#define LIBTOPOMAP_HPP

//#define BGP

#define DBG5(stmt) stmt // debug TPM_Benchmark
#define DBG4(stmt) // debug greedy mapping
#define DBG3(stmt) // debug get_cong
#define DBG2(stmt) // debug startup and initial graph distribution
#define DBG1(stmt)  // debug SSSP
#define DBG(stmt) // debug parser
#define MEM(stmt) stmt // print memory consumption information for the graphs
#define INFO(stmt) stmt // print additional information about optimization

// rcm-specifics
#define RCM_PREFIX TPM_
#include "rcm.h"

// enable all sorts of (unnecessary) checks (for safety)
#define CHECKS 1

typedef std::vector<std::string> TPM_Nodenames;
typedef std::vector<std::vector<int> > TPM_Graph; // adjacency list graph representation
typedef std::vector<std::vector<int> > TPM_Graph_ewgts; // edge weights
typedef std::vector<std::vector<int64_t> > TPM_Graph_lewgts; // long edge weights
typedef std::vector<std::vector<double> > TPM_Graph_dewgts; // double edge weights
typedef std::vector<int> TPM_Graph_vwgts; // vertex weights
typedef std::vector<double> TPM_Graph_dists; // SSSP dists
typedef std::vector<int> TPM_Graph_preds; // SSSP predecessors
typedef std::vector<int> TPM_Mapping; // mapping (permutation)

// this is exported and NULL by default but can be set to a filename to
// read the fake mapping (rank - fake-name) from :-)
extern char *TPM_Fake_names_file; 

class dijk_compare_func {
  public:
  bool operator()(std::pair<double,int> x, std::pair<double,int> y) {
    if(x.first > y.first) return true;
    return false;
  }
};

class max_compare_func {
  public:
  bool operator()(std::pair<double,int> x, std::pair<double,int> y) {
    if(x.first < y.first) return true;
    return false;
  }
};

// in topoparser.cpp
int TPM_Read_topo(const char *topofile, TPM_Nodenames *names, TPM_Graph *g, TPM_Graph_ewgts *w);
int TPM_Fake_hostname(const char *fname, int rank, char *name, int namesize);

// in libtopomap.cpp
int TPM_Topomap_greedy(MPI_Comm distgr, const char *topofile, int max_mem, int *newrank);
int TPM_Topomap(MPI_Comm distgr, const char *topofile, int max_mem, int *newrank);

// in bgbtopo.cpp
int TPM_Get_bgp_topo(int r, TPM_Graph *g, int *me);

// tools in libtopomap.cpp (should go somewhere else)
int TPM_Write_graph_comm(MPI_Comm grphcomm, const char *filename);
int TPM_Write_phystopo(MPI_Comm grphcomm, const char *filename, const char *topofile);
void TPM_Benchmark_graphtopo(MPI_Comm distgr, int newrank, int dsize, double *before, double *after);

// on-node mapping
int TPM_Node_mapping(TPM_Graph* ltg_ref, TPM_Graph_ewgts *ltgw_ref, TPM_Mapping* idx2rank_ref, MPI_Comm comm, int rank);

#endif
