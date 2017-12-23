#include "libtopomap.hpp"
#include <parmetis.h>
#include <metis.h>
#ifdef HAVE_SCOTCH
#include <scotch.h>
#endif /* HAVE_SCOTCH */
#include "MersenneTwister.h"

extern "C" {
  void METIS_PartGraphKway(int*, idxtype*, idxtype*, idxtype*, idxtype*, int*, int*, int*, int*, int*, idxtype*);

#ifdef HAVE_SCOTCH
  int                         SCOTCH_archInit     (SCOTCH_Arch * const);
  void                        SCOTCH_archExit     (SCOTCH_Arch * const);
  int                         SCOTCH_archBuild    (SCOTCH_Arch * const, const SCOTCH_Graph * const, const SCOTCH_Num, const SCOTCH_Num * const, const SCOTCH_Strat * const);
  int                         SCOTCH_graphInit    (SCOTCH_Graph * const);
  void                        SCOTCH_graphExit    (SCOTCH_Graph * const);
  int                         SCOTCH_graphMap     (SCOTCH_Graph * const, const SCOTCH_Arch * const, SCOTCH_Strat * const, SCOTCH_Num * const);
  int                         SCOTCH_stratInit    (SCOTCH_Strat * const);
  void                        SCOTCH_stratExit    (SCOTCH_Strat * const);
#endif
}

typedef struct {
    double a;
    int b;
} doubleint;

// this is an exported variable that can be set in order to use the fake
// names file ;)
char *TPM_Fake_names_file = NULL;

/* this implements the Dijkstra algorithm */
static void TPM_SSSP(TPM_Graph *g_ref, TPM_Graph_dewgts *w_ref, int src, TPM_Graph_dists *dist_ref, TPM_Graph_preds *pred_ref) {
  TPM_Graph& g= *g_ref; // the graph!
  TPM_Graph_dewgts& w= *w_ref; // edge weights
  TPM_Graph_dists& dist= *dist_ref; // distance vector
  TPM_Graph_preds& pred = *pred_ref; // predecessor map

  std::vector<bool> finished(g.size(),false);

  // infinite initial distance and invalid predecessor
  std::fill(dist.begin(), dist.end(), std::numeric_limits<double>::max());
  std::fill(pred.begin(), pred.end(), -1);

  // Dijkstra queue
  std::priority_queue<std::pair<double /* dist */,int /* index */>, std::vector<std::pair<double,int> >, dijk_compare_func > Q;
  std::pair <double,int> Qel;

  dist[src] = 0;
  Q.push(std::pair <double,int> (dist[src], src));

  while(!Q.empty()) {
    // take the cheapest vertex out of the queue
    Qel = Q.top();
    Q.pop();
    int u = Qel.second;
    DBG1(printf("finished vertex %i at dist %.2f\n", u, dist[u]));
    // we need to do this because we can't change elements in the queue, thus, we just throw 
    // more in every time we find a shorter path (so each vertex might be more than once in the queue)
    if(!finished[u]) { 
      finished[u]=true;
      // investigate all neighbors of current vertex u
      for(int i=0; i<g[u].size(); ++i) {
        int v = g[u][i];
        double vweight = w[u][i];
        // did we discover a shorter route?
        if(!finished[v] && dist[u] + vweight < dist[v]) {
          dist[v] = dist[u] + vweight; 
          pred[v] = u;
          DBG1(printf("discovered path to %i and dist %.2f\n", v, dist[v]));
          Q.push(std::pair <double,int> (dist[v], v));
        }
      }
    }
  }
}

static void TPM_Get_cong(int r, TPM_Graph *h_ref, TPM_Graph_ewgts *ptgc_ref, TPM_Graph *g_ref, TPM_Graph_ewgts *weights_ref, TPM_Mapping *map_ref, double *max_cong, int *max_dil, bool print=false, double *avg_cong=NULL, double *avg_dil=NULL, const char *outfile=NULL) {
  TPM_Graph_ewgts& ltgw= *weights_ref; // ltg edge weights!
  TPM_Graph_ewgts& ptgc= *ptgc_ref; // ptg capacity weights!
  TPM_Mapping& mapping = *map_ref; // mapping from vertex in g to vertex in h
  TPM_Graph& g= *g_ref; // embedded graph
  TPM_Graph& h= *h_ref; // host graph

  TPM_Graph_dewgts w(h.size());
  // init edge weights in h to n*n to force shortest paths
  int maxw=0; 
  assert(g.size() == ltgw.size());
  for(int i=0; i<g.size(); ++i) {
    if(ltgw[i].size() != 0) {
      //printf("[%i] ... %i %i %x %i\n", r, i, ltgw.size(), &ltgw[i][0], ltgw[i].size());
      maxw=std::max(maxw, *std::max_element(ltgw[i].begin(), ltgw[i].end()) );
    }
  }
  const double init=maxw*h.size()*h.size();
  assert(init>0); // check for overflow
  for(int i=0; i<h.size(); ++i) {
    w[i].resize(h[i].size());
    std::fill(w[i].begin(), w[i].end(), init);
  }

  // for each vertex, annotate the edges on shortest path
  // this would simulate a routing which ideally balances packets along
  // shortest paths !
  // Oblivious static routing would cause more congestion!
  *max_cong=*max_dil=0;
  TPM_Graph_dists dist(h.size());
  TPM_Graph_preds pred(h.size());
  int g_edges=0, dil_sum=0; // to compute average dilation
  // iterate over all vertices in logical topology (p)
  for(int i=0; i<g.size(); i++) {
    // iterate over all neighbors of each vertex in topology
    for(int j=0; j<g[i].size(); j++) {
      // find vertices in physical topology
      int src = mapping[i];
      int tgt = mapping[g[i][j]];

      if(print) printf("checking virtual topo edge %i -> %i (mapped %i -> %i)\n", i, g[i][j], src, tgt);
      // do dijkstra to find shortest path
      TPM_SSSP(&h, &w, src, &dist, &pred);

      // walk predecessor array backwards
      int v=tgt;
      int dilation = 0; // two cables at least per hop but one is already in ltg but if we map on multicore, the average dilation can become negative :-/ -- thus init to 0
      double maxc=0;
//if(!r) printf("--- tgt: %i, src: %i\n", tgt, src);
      while(v != src) {
        dilation++;
        // find position of edge pred[v] -> v in edge weights map (same
        // as in h -- thus, find position in h :)
        assert(h.size() > pred[v]);
        int pos; for(pos=0; h[pred[v]][pos] != v; pos++) if(pos >= h[pred[v]].size()) { printf("[%i] pred %i not found for %i\n", r, pred[v], v); break; }
        assert(w.size() > pred[v]);
        assert(w[pred[v]].size() > pos);
//if(!r) printf(".. step: %i cost: %.2f\n", v, w[pred[v]][pos]);

        w[pred[v]][pos] += (double)ltgw[i][j]/ptgc[pred[v]][pos]; // add weight of logical edge divided by capacity of physical edge
        maxc=std::max(w[pred[v]][pos] - init, maxc);

        if(print) printf("increasing edge weight on edge %i -> %i to %i\n", pred[v], v, w[pred[v]][pos]-init);
        v = pred[v];
      }
      *max_cong = std::max(maxc,*max_cong);
      *max_dil = std::max(dilation,*max_dil);
      
      if(avg_dil != NULL) {
        g_edges++;
        dil_sum+=dilation;
      }

      DBG3(printf("%i -> %i: %.2f, %i\n", src, tgt, *max_cong, dilation));
    }
  }
  // at this point, we have fully annotated the graph and might want
  // to loop over it again to get the congestion sum, however, for the
  // congestion max, we don't need to do this!
  if(avg_cong!=NULL) {
    int h_used_edges=0;
    double h_cong_sum=0;
    for(int i=0; i<h.size(); ++i) {
      for(int j=0; j<h[i].size(); ++j) {
        if(w[i][j] > init) {
          h_used_edges++;
          h_cong_sum+=(w[i][j]-init);
        }
      }
    }
    *avg_cong = h_cong_sum/h_used_edges;
  }
  if(avg_dil != NULL) *avg_dil = (double)dil_sum/g_edges;
  
  /* this section only writes the graph if necessary */
  if(r == 0 && outfile != NULL) {
    FILE *f = fopen(outfile, "w");
    fprintf(f, "digraph net {\n");
    
    // print mapping 
    for(int i=0; i<mapping.size(); ++i) fprintf(f, "%i [ rank=\"%i\" ]\n", mapping[i], i);

    for(int i=0; i<h.size(); ++i) {
      for(int j=0; j<h[i].size(); ++j) {
        // only print edges that carry some weight (they don't have to
        // be mapped if they're routed through ...)
        if(w[i][j] > init) fprintf(f, "%i -> %i [ weight=\"%f\" ]\n", i, h[i][j], w[i][j]-init);
      }
    }
    fprintf(f, "}\n");
    fclose(f);
    INFO(printf("[%s written]\n", outfile));
  }
  /* end of graph output section */
}

static int TPM_Get_myid(int r, TPM_Nodenames *names) {
  // determine my hostname
  const int len=1024;
  char name[len];
  if(TPM_Fake_names_file != NULL) {
    TPM_Fake_hostname(TPM_Fake_names_file, r, name, len);
  } else {
    // determine my hostname
    gethostname(name, len);
  }
  // scan for and remove first "."
  for(int i=0; i<len; ++i) if(name[i] == '.') { name[i] = '\0'; break; }
  std::string hostname = name;

  // find my vertex-id in graph -- linear search (but is only done once,
  // could turn name-table into a map!
  int me=-1;
  for(int i=0; i<names->size(); ++i) if((*names)[i] == hostname) { me=i; break; }
  if(me == -1) {
    printf("[%i] didn'n find my own hostname (%s) in nametable! Aborting job.\n", r, hostname.c_str());
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  DBG2(printf("[%i] found myself (%s) as vertex %i!\n", r, hostname.c_str(), me));

  return me;
}

/* uses the reverse Cuthill Mckee algorithm top optimize the mapping.
 * This approach applies rcm to the topology graph as well as the
 * problem graph and then maps them together. This of course only works
 * if both graphs have the same cardinality! */
int TPM_Map_rcm(int r, TPM_Graph *ptg_ref, TPM_Graph_vwgts *nprocs_ref, TPM_Graph *ltg_ref, TPM_Mapping *mapping_ref, int start) {
  // turn refs into normal objects
  TPM_Graph& ptg = *ptg_ref;
  TPM_Graph& ltg = *ltg_ref;
  TPM_Graph_vwgts& nprocs = *nprocs_ref;
  TPM_Mapping& mapping = *mapping_ref;

  //MPI_Barrier(MPI_COMM_WORLD);

  // if cardinalities don't match
  TPM_Graph rtg;
  TPM_Mapping rtg2ptgmap(ptg.size()); // map to translate rtg vertices to ptg vertices
  int use_rtg = 0; // decide which graph to use for RCM of ptg
  if(ptg.size()!=ltg.size()) {
    // figure out which subgraph is actually used ... gaah, it is a
    // Steiner problem and the rcm doesn't quite work as it seems :-(
    // as a simple fix, just remove all unused (nprocs == 0) vertices (and
    // all neighboring edges) from ptg! This might disconnect ptg but RCM
    // does not care :-)!
    assert(ptg.size() == nprocs.size());
    // to remove from ptg, we build a new graph called rtg (RCM Topology Graph)
    TPM_Mapping rtgmap(ptg.size(), -1); // new numbers in rtg 
    int cnt=0;
    for(int i=0; i<ptg.size(); ++i) {
      if(nprocs[i] != 0) {
        rtgmap[i] = cnt; 
        rtg2ptgmap[cnt] = i;
        cnt++;
      }
    }

    for(int i=0; i<ptg.size(); ++i) if(nprocs[i] != 0) {
      // translate all adjacent vertices!
      std::vector<int> adj;
      for(int j=0; j<ptg[i].size(); ++j) {
        int vert = ptg[i][j];
        if(rtgmap[j] != -1) adj.push_back(rtgmap[j]);
      }
      rtg.push_back(adj);
    }
    use_rtg = 1;
  }

  // there should not be more than one proc in nprocs
#ifdef CHECKS
  for(int i=0; i<nprocs.size(); ++i) if(nprocs[i]>1)  {
    printf("[%i] nprocs at pos %i > 1 (%i)!\n", r, i, nprocs[i]);
    MPI_Abort(MPI_COMM_WORLD, 3);
  }
#endif
  
  std::vector<int> rcm_ptg_map;
  if(use_rtg) {
    rcm_ptg_map.resize(rtg.size());
    int rtgn=0; for(int i=0; i<rtg.size(); ++i) rtgn+=rtg[i].size();
    std::vector<idxtype>  xadj(rtg.size()+1), // CSR index 
                          adjncy(rtgn); // CSR list
    for(int i=0; i<(rtg.size()+1); i++) if(i==0) xadj[i]=0; else xadj[i]=xadj[i-1]+rtg[i-1].size();
    int pos=0; for(int i=0; i<rtg.size(); i++) for(int j=0; j<rtg[i].size(); ++j) { adjncy[pos++] = rtg[i][j]; assert(rtg[i][j]<rtg.size()); }

    std::vector<signed char> mask(rtg.size(), 1);
    std::vector<int> degs(rtg.size());
    TPM_genrcmi(rtg.size(), 0, &xadj[0], &adjncy[0], &rcm_ptg_map[0], &mask[0], &degs[0]);
  } else { 
    rcm_ptg_map.resize(ptg.size());
    int ptgn=0; for(int i=0; i<ptg.size(); ++i) ptgn+=ptg[i].size();
    std::vector<idxtype>  xadj(ptg.size()+1), // CSR index 
                          adjncy(ptgn); // CSR list
    for(int i=0; i<(ptg.size()+1); i++) if(i==0) xadj[i]=0; else xadj[i]=xadj[i-1]+ptg[i-1].size();
    int pos=0; for(int i=0; i<ptg.size(); i++) for(int j=0; j<ptg[i].size(); ++j) { adjncy[pos++] = ptg[i][j]; assert(ptg[i][j]<ptg.size()); }

    std::vector<signed char> mask(ptg.size(), 1);
    std::vector<int> degs(ptg.size());
    TPM_genrcmi(ptg.size(), 0, &xadj[0], &adjncy[0], &rcm_ptg_map[0], &mask[0], &degs[0]);
  }

  //if(!r) { printf("ptg map: "); for(int i=0; i<rcm_ptg_map.size(); ++i) printf("%i ", rcm_ptg_map[i]); printf("\n"); }

  std::vector<int> rcm_ltg_map(ltg.size());
  {
    int ltgn=0; for(int i=0; i<ltg.size(); ++i) ltgn+=ltg[i].size();
    std::vector<idxtype>  xadj(ltg.size()+1), // CSR index 
                          adjncy(ltgn); // CSR list
    for(int i=0; i<(ltg.size()+1); i++) if(i==0) xadj[i]=0; else xadj[i]=xadj[i-1]+ltg[i-1].size();
    int pos=0; for(int i=0; i<ltg.size(); i++) for(int j=0; j<ltg[i].size(); ++j) { adjncy[pos++] = ltg[i][j]; assert(ltg[i][j]<ltg.size()); }

    std::vector<signed char> mask(ltg.size(), 1);
    std::vector<int> degs(ltg.size());
    TPM_genrcmi(ltg.size(), 0, &xadj[0], &adjncy[0], &rcm_ltg_map[0], &mask[0], &degs[0]);
  }

  //if(!r) { printf("ltg map: "); for(int i=0; i<rcm_ltg_map.size(); ++i) printf("%i ", rcm_ltg_map[i]); printf("\n"); }

  // translate mappings
  if(use_rtg) {
    for(int i=0; i<ltg.size(); ++i) mapping[rcm_ltg_map[i]] = rtg2ptgmap[rcm_ptg_map[i]]; // TODO: not 100% sure that this is right
  } else {
    for(int i=0; i<ltg.size(); ++i) mapping[rcm_ltg_map[i]] = rcm_ptg_map[i]; // TODO: not sure either
  }

  //if(!r) { printf("final mapping: "); for(int i=0; i<mapping.size(); ++i) printf("%i ", mapping[i]); printf("\n"); }
  return 0;
}

/* this is the recursive mapping approach similar to SCOTCH */
static int TPM_Map_recursive(int r, TPM_Graph *ptg_ref, TPM_Graph_ewgts *ptgc_ref, 
    TPM_Graph_vwgts *nprocs_ref, TPM_Graph *ltg_ref, TPM_Graph_ewgts *weights_ref, 
    TPM_Mapping *mapping_ref, TPM_Mapping ltg_map /* indices in logical topology */, TPM_Mapping *ptg_map_ref /* map from passed ptg (subgraph) into global ptg */) {

  // turn refs into normal objects
  TPM_Graph& ptg = *ptg_ref;
  TPM_Graph& ptgc = *ptgc_ref; // ptg edge capacities!
  TPM_Graph& ltg = *ltg_ref;
  TPM_Graph& ltgw = *weights_ref; // ltg edge weights!
  TPM_Graph_vwgts& nprocs = *nprocs_ref;
  TPM_Mapping& mapping = *mapping_ref;
  TPM_Mapping& pmap= *ptg_map_ref;

  //if(!r) printf("[%i] TPM_MAP_RECURSIVE: nprocs: %i, ptg: %i ltg: %i\n", r, std::accumulate(nprocs.begin(), nprocs.end(), 0), ptg.size(), ltg_map.size());
  
  if(ltg_map.size() != 1) {

    //if(r) MPI_Barrier(MPI_COMM_WORLD);

    int n = ptg.size();
    int nedges=0; for(int i=0; i<ptg.size(); ++i) nedges += ptg[i].size(); 
    
    std::vector<idxtype>  xadj(n+1),
                          adjncy(nedges+1),
                          part(n),
                          vwgt(n+1),
                          adjwgt(nedges+1);

    int wgtflag = 2; // edge and vertex weights weights 
    int numflag = 0;
    int nparts = 2;
    int options[5] = {0,0,0,0,0};
    int edgecut = 0;

    int offset = 0;
    int adjoff = 0;
    for(int i=0; i<n; ++i) {
      xadj[i] = offset;
      offset += ptg[i].size();

      vwgt[i] = nprocs[i];

      //if(!r) { if(i==0) printf("nprocs: "); if(vwgt[i] != 0) printf("%i ", i); if(i==n-1) printf("\n"); }

      //if(!r) printf("|(off: %i, wgt: %i) ", xadj[i], vwgt[i]);
      for(int j=0; j<ptg[i].size(); ++j) {
        adjncy[adjoff] = ptg[i][j];
        adjwgt[adjoff] = ptgc[i][j];
        //if(!r) printf("%i ", adjncy[adjoff]);
        adjoff++;
      }
    }
    xadj[n] = offset;
    //if(!r) printf("|(%i)\n", xadj[n]);

    METIS_PartGraphRecursive(&n, &xadj[0], &adjncy[0], &vwgt[0], &adjwgt[0], &wgtflag, &numflag, &nparts, &options[0], &edgecut, &part[0]);

    int group1=0, group0=0, diff=2;

    while(diff > 1) {
      group0=0;
      group1=0;
      for(int i=0; i<n; ++i) {
        if(part[i] == 0) group0 += nprocs[i];
        if(part[i] == 1) group1 += nprocs[i];
      }
      // printf("[%i] pmap group0: %i, group1: %i\n", r, group0, group1);
      // check if it's balanced or not -- correct if necessary
      diff = group0-group1;
      if(diff < 0) diff *= -1;
      if(diff > 1) { //it's not balanced!
        int min, min_vert;
        if(group0 > group1) { // reset one 0 to a 1 in part: pick the least connected one!
          min=std::numeric_limits<int>::max();
          min_vert=-1;
          for(int i=0; i<part.size(); ++i) {
            if(part[i] == 0 && nprocs[i] <= diff) {
              int outweight = std::accumulate(ptgc[i].begin(), ptgc[i].end(), 0);
              if(min > outweight) {
                min_vert = i;
                min = outweight;
              }
            }
          }
          assert(min_vert != -1);
          part[min_vert] = 1;
        }
        if(group1 > group0) { // reset one 1 to a 0 in part: pick the least connected one!
          min=std::numeric_limits<int>::max();
          min_vert=-1;
          for(int i=0; i<part.size(); ++i) {
            if(part[i] == 1 && nprocs[i] <= diff) {
              int outweight = std::accumulate(ptgc[i].begin(), ptgc[i].end(), 0);
              if(min > outweight) {
                min_vert = i;
                min = outweight;
              }
            }
          }
          assert(min_vert != -1);
          part[min_vert] = 0;
        }
        if(!r) printf("correcting pmap partition ... moving vert %i with minimal weight %i\n", min_vert, min);
      }
    }

    TPM_Graph_vwgts nprocs0, nprocs1;
    //TPM_Graph_vwgts nprocs0(n,0), nprocs1(n,0);
    // fill new Mapping vectors 
    /*for(int i=0; i<n; ++i) {
      // fill only nprocs in the right partitions!
      if(part[i] == 0) {
        nprocs0[i] = nprocs[i];
      } else {
        nprocs1[i] = nprocs[i];
      }
    }*/

    // remove all cut edges from ptg
    /*for(int i=0; i<ptg.size(); ++i) {
      for(int j=0; j<ptg[i].size(); ) {
        if(part[i] != part[ ptg[i][j] ]) {
          if(!r) printf("removed edge %i->%i (%i != %i)\n", i, ptg[i][j], part[i], part[ ptg[i][j] ]);
          ptg[i].erase(ptg[i].begin()+j);
          ptgc[i].erase(ptgc[i].begin()+j);
        } else j++;
      }
    }*/

    // build two ptgs for the two partitions 
    TPM_Graph ptg0, ptg1;
    TPM_Graph_ewgts ptgc0, ptgc1;
    TPM_Mapping pmap0, pmap1; // needed to translate ranks back to original graph

    TPM_Mapping newverts(ptg.size()); // translation table from old vertices to index in two new graphs
    int idx0=0, idx1=0;
    for(int i=0; i<ptg.size(); ++i) {
      if(part[i] == 0) { newverts[i] = idx0++; }
      if(part[i] == 1) { newverts[i] = idx1++; }
    }

    for(int i=0; i<part.size(); ++i) {
      if(part[i] == 0) {
        ptg0.resize(ptg0.size()+1);
        ptgc0.resize(ptgc0.size()+1);
        nprocs0.push_back(nprocs[i]);
        pmap0.push_back(pmap[i]);
        for(int j=0; j<ptg[i].size(); ++j) {
          // only add edges that are leading to a vertex in the same partition
          if(part[i] == part[ ptg[i][j] ]) {
            ptg0[ptg0.size()-1].push_back(newverts[ptg[i][j]]);
            ptgc0[ptgc0.size()-1].push_back(ptgc[i][j]);
          }
        }
      }
      if(part[i] == 1) {
        ptg1.resize(ptg1.size()+1);
        ptgc1.resize(ptgc1.size()+1);
        nprocs1.push_back(nprocs[i]);
        pmap1.push_back(pmap[i]);
        for(int j=0; j<ptg[i].size(); ++j) {
          // only add edges that are leading to a vertex in the same partition
          if(part[i] == part[ ptg[i][j] ]) {
            ptg1[ptg1.size()-1].push_back(newverts[ptg[i][j]]);
            ptgc1[ptgc1.size()-1].push_back(ptgc[i][j]);
          }
        }
      }
    }

    /*if(!r) {
      printf("ptg group0 (edgecut: %i): ", edgecut);
      for(int i=0; i<nprocs0.size(); ++i) if(nprocs0[i] != 0) printf(" %i ", pmap[i]);
      printf("\n");
    }*/

    /******* partition logical topology graph ********/
    n = ltg_map.size();
    nedges=0; for(int i=0; i<ltg_map.size(); ++i) nedges += ltg[ ltg_map[i] ].size(); 

    // invert ptg_map (ptg graph) to translate adjacency lists to new graph
    int max = *std::max_element(ltg_map.begin(),ltg_map.end()); 
    std::vector<idxtype> invmap(max+1,-1);
    for(int i=0; i<n; ++i) invmap[ ltg_map[i] ] = i;
    // end inversion
    
    xadj.resize(n+1);
    vwgt.resize(n+1);
    adjncy.resize(nedges+1);
    adjwgt.resize(nedges+1);
    part.resize(n);

    wgtflag = 1; // only edgeweights
    numflag = 0;
    nparts = 2;
    edgecut = 0;

    offset = 0;
    adjoff = 0;
    for(int i=0; i<n; ++i) {
      xadj[i] = offset;
      // compute the right offset (only including edges in our group)
      for(int j=0; j<ltg[ ltg_map[i] ].size(); ++j) if(ltg[ ltg_map[i] ][j] <= max && invmap[ ltg[ ltg_map[i] ][j] ] != -1) offset++;

      vwgt[i] = 1;

      for(int j=0; j<ltg[ ltg_map[i] ].size(); ++j) {
        // see ptg for comments about datastructures
        int dst = -1;
        if(ltg[ ltg_map[i] ][j] <= max) { // if the destination is > max then it's not in our set!
          dst = invmap[ ltg[ ltg_map[i] ][j] ];
        }
        // -1 are the destinations outside our group! Exclude them!
        if(dst != -1) {
          adjncy[adjoff] = dst;
          adjwgt[adjoff] = ltgw[ ltg_map[i] ][j];
          adjoff++;
          //printf("i: %i, ptg_map[i]: %i, j: %i, ptg[ ptg_map[i] ][j]: %i, invmap[ ptg[ ptg_map[i] ][j] ]: %i\n", i, ptg_map[i], j, ptg[ ptg_map[i] ][j], invmap[ ptg[ ptg_map[i] ][j] ]);
        }
      }
    }
    xadj[n] = offset;

    //if(!r) printf("calling second Metis\n");
    METIS_PartGraphRecursive(&n, &xadj[0], &adjncy[0], NULL, &adjwgt[0], &wgtflag, &numflag, &nparts, &options[0], &edgecut, &part[0]);

    //if(!r) printf("[%i] lmap group0: %i, group1: %i\n", r, group0, group1);

    // check if it's balanced or not -- correct if necessary
    diff = 2; // enter at least once :)
    while(diff > 1) {
      group1 = std::accumulate(part.begin(), part.end(), 0); // number of vertices in the second group 
      group0 = part.size()-group1;
      diff = group0-group1;
      if(diff < 0) diff *= -1;
      if(diff > 1) { //it's not balanced!
        int min_vert, min;
        if(group0 > group1) { // reset one 0 to a 1 in part: pick the least connected one!
          min=std::numeric_limits<int>::max();
          min_vert=-1;
          for(int i=0; i<part.size(); ++i) {
            if(part[i] == 0) {
              int outweight = std::accumulate(ltgw[ ltg_map[i] ].begin(), ltgw[ ltg_map[i] ].end(), 0);
              if(min > outweight) {
                min_vert = i;
                min = outweight;
              }
            }
          }
          assert(min_vert != -1);
          part[min_vert] = 1;
        }
        if(group1 > group0) { // reset one 0 to a 1 in part: pick the least connected one!
          min=std::numeric_limits<int>::max();
          min_vert=-1;
          for(int i=0; i<part.size(); ++i) {
            if(part[i] == 1) {
              int outweight = std::accumulate(ltgw[ ltg_map[i] ].begin(), ltgw[ ltg_map[i] ].end(), 0);
              if(min > outweight) {
                min_vert = i;
                min = outweight;
              }
            }
          }
          assert(min_vert != -1);
          part[min_vert] = 0;
        }
        if(!r) printf("correcting lmap partition ... moving vert %i with minimal weight %i (group0: %i, group1: %i)\n", min_vert, min, group0, group1);
      }
    }

    TPM_Mapping lmap1, lmap2;
    // fill new Mapping vectors 
    for(int i=0; i<part.size(); ++i) {
      //if(!r) printf("%i\n", part[i]);

      // the i-th entry in part is the i-th entry in ptg_map
      if(part[i] == 0) {
        lmap1.push_back( ltg_map[i] );
      } else {
        lmap2.push_back( ltg_map[i] );
      }
    }

    /*if(!r) {
      printf("ltg group0 (edgecut: %i): ", edgecut);
      for(int i=0; i<lmap1.size(); ++i) printf(" %i ", lmap1[i]);
      printf("\n");
    }*/

    // save memory in recursion!
    xadj.resize(0);
    vwgt.resize(0);
    adjncy.resize(0);
    adjwgt.resize(0);
    part.resize(0);
    invmap.resize(0);
    
    //if(!r) printf("calling recursively %i %i, %i %i\n", lmap1.size(), pmap1.size(), lmap2.size(), pmap2.size());
    // call recursively (two subtrees)
   // check which pmap fits which lmap size (might be two different if uneven group size) :-) 
   int nprocs0_size = std::accumulate(nprocs0.begin(), nprocs0.end(), 0);
   int nprocs1_size = std::accumulate(nprocs1.begin(), nprocs1.end(), 0);

   if(lmap1.size() == nprocs0_size && lmap2.size() == nprocs1_size) { 
      TPM_Map_recursive(r, &ptg0, &ptgc0, &nprocs0, ltg_ref, weights_ref, mapping_ref, lmap1, &pmap0);
      TPM_Map_recursive(r, &ptg1, &ptgc1, &nprocs1, ltg_ref, weights_ref, mapping_ref, lmap2, &pmap1);
   } else if(lmap1.size() == nprocs1_size && lmap2.size() == nprocs0_size) {
      TPM_Map_recursive(r, &ptg1, &ptgc1, &nprocs1, ltg_ref, weights_ref, mapping_ref, lmap1, &pmap1);
      TPM_Map_recursive(r, &ptg0, &ptgc0, &nprocs0, ltg_ref, weights_ref, mapping_ref, lmap2, &pmap0);
   } else {
     if(!r) printf("[%i] mapping sizes don't match (lmap1.size(): %i, nprocs1_size: %i, lmap2.size(): %i, nprocs0_size: %i) -- this should never happen - abort!\n", r, lmap1.size(), nprocs1_size, lmap2.size(), nprocs0_size);
     MPI_Barrier(MPI_COMM_WORLD);
     MPI_Abort(MPI_COMM_WORLD, 4);
   }
    
  } else {
    // end of recursion -- finalize mapping
    assert(std::accumulate(nprocs.begin(), nprocs.end(), 0) == 1); // only one nproc in ptg left
    int index = 0; while(nprocs[index] == 0) index++;

    assert(ltg_map.size() == 1);
    mapping[ ltg_map[0] ] = pmap[index];

    //if(!r) printf("mapping: %i to %i (index: %i)\n", pmap[index], ltg_map[0], index);
  }
  return 0;
}

static int TPM_Map_scotch(int r, TPM_Graph *ptg_ref, TPM_Graph_ewgts *ptgc_ref, TPM_Graph_vwgts *nprocs_ref, TPM_Graph *ltg_ref, TPM_Graph_ewgts *weights_ref, TPM_Mapping *mapping_ref, int start) {
  // turn refs into normal objects
  TPM_Graph& ptg = *ptg_ref; // the physical graph topology as adjacency list
  TPM_Graph& ptgc = *ptgc_ref; // the edge capacities in the physical graph topology
  TPM_Graph& ltg = *ltg_ref; // the logical graph topology as adjacency list
  TPM_Graph& ltgw = *weights_ref; // the edge weights in the logical graph topology
  TPM_Graph_vwgts& nprocs = *nprocs_ref; // the number of CPUs per node in the physical topology (enables heterogeneous systems)
  TPM_Mapping& mapping = *mapping_ref; // the mapping (output)

#ifdef HAVE_SCOTCH
  SCOTCH_Strat        logsdat;                    // Graph mapping strategy (default of its kind)
  SCOTCH_Strat        physdat;                    // Architecture building strategy (default of its kind)
  SCOTCH_Arch         phyadat;                    // Physical graph architecture
  SCOTCH_Graph        phygdat;                    // Physical graph data
  SCOTCH_Graph        loggdat;                    // Logical graph

  SCOTCH_archInit  (&phyadat);                    // Initialize architecture structure

  int edgenbr=0;
  for(int i=0; i<ptg.size(); ++i) edgenbr += ptg[i].size();
  std::vector<int> edgetab(edgenbr);
  std::vector<int> edlotab(edgenbr);
  std::vector<int> verttab(ptg.size()+1);
  int iter=0;
  verttab[0] = 0;
  for(int i=0; i<ptg.size(); ++i) {
    std::copy(ptg[i].begin(), ptg[i].end(), &edgetab[iter]);
    iter += ptg[i].size();
    verttab[i+1]=iter;
  }

  {
    assert(ptg.size() == ptgc.size());
    iter=0;
    for(int i=0; i<ptgc.size(); ++i) { assert(ptg[i].size() == ptgc[i].size()); std::copy(ptgc[i].begin(), ptgc[i].end(), &edlotab[iter]); iter += ptgc[i].size(); }

    SCOTCH_graphInit  (&phygdat);

    /*
    printf("verttab (%i): ", ptg.size());
    for(int i=0; i<verttab.size(); ++i) { printf("%i ", verttab[i]); } printf("\n");
    printf("edgetab (%i): ", edgenbr);
    for(int i=0; i<edgetab.size(); ++i) { printf("%i ", edgetab[i]); } printf("\n");
    printf("edlotab: ");
    for(int i=0; i<edlotab.size(); ++i) { printf("%i ", edlotab[i]); } printf("\n");*/

    SCOTCH_graphBuild (&phygdat, 
      0, // const SCOTCH_Num            baseval,              /* Base value                          */
      ptg.size(), //const SCOTCH_Num            vertnbr,              /* Number of vertices                  */
      &verttab[0], // const SCOTCH_Num * const    verttab,              /* Vertex array [vertnbr or vertnbr+3] */
      NULL, // const SCOTCH_Num * const    vendtab,              /* Vertex end array [vertnbr]          */
      &nprocs[0], // const SCOTCH_Num * const    velotab,              /* Vertex load array                   */
      NULL, // const SCOTCH_Num * const    vlbltab,              /* Vertex label array                  */
      edgenbr, // const SCOTCH_Num            edgenbr,              /* Number of edges (arcs)              */
      &edgetab[0], // const SCOTCH_Num * const    edgetab,              /* Edge array [edgenbr]                */
      &edlotab[0]); // const SCOTCH_Num * const    edlotab);              /* Edge load array                     */
    //assert (SCOTCH_graphCheck (&phygdat) == 0);
  }
// provide list of to-be-mapped to vertices and size ...
  assert(nprocs.size() == ptg.size());
  std::vector<int> mapping_nodes;
  for(int i=0; i<ptg.size(); ++i) if(nprocs[i] > 0) { mapping_nodes.push_back(i); }

  //printf("mapping_nodes: "); for(int i=0; i<mapping_nodes.size(); ++i) { printf("%i ", mapping_nodes[i]); } printf("\n");

  SCOTCH_stratInit (&physdat);
  SCOTCH_archBuild (&phyadat, &phygdat, mapping_nodes.size(), &mapping_nodes[0], &physdat); // Compute target architecture from physical graph
  SCOTCH_stratExit (&physdat);                    // Free architecture recursive bipartitioning strategy
  SCOTCH_graphExit (&phygdat);                    // Free architecture graph data (vectors can be safely re-used)

  edgenbr=0;
  for(int i=0; i<ltg.size(); ++i) edgenbr += ltg[i].size();
  edgetab.resize(edgenbr);
  edlotab.resize(edgenbr);
  verttab.resize(ltg.size()+1);

  {
    int iter=0;
    verttab[0] = 0;
    for(int i=0; i<ltg.size(); ++i) { std::copy(ltg[i].begin(), ltg[i].end(), &edgetab[iter]); iter += ltg[i].size(); verttab[i+1]=iter; } 
    iter=0;
    assert(ltg.size() == ltgw.size());
    for(int i=0; i<ltgw.size(); ++i) { std::copy(ltgw[i].begin(), ltgw[i].end(), &edlotab[iter]); iter += ltgw[i].size(); }

    SCOTCH_graphInit (&loggdat);                    // Initialize logical graph structure
    SCOTCH_graphBuild (&loggdat, 
      0, // const SCOTCH_Num            baseval,              /* Base value                          */
      ltg.size(), // const SCOTCH_Num            vertnbr,              /* Number of vertices                  */
      &verttab[0], // const SCOTCH_Num * const    verttab,              /* Vertex array [vertnbr or vertnbr+1] */
      NULL, // const SCOTCH_Num * const    vendtab,              /* Vertex end array [vertnbr]          */
      NULL, // const SCOTCH_Num * const    velotab,              /* Vertex load array                   */
      NULL, // const SCOTCH_Num * const    vlbltab,              /* Vertex label array                  */
      edgenbr, // const SCOTCH_Num            edgenbr,              /* Number of edges (arcs)              */
      &edgetab[0], // const SCOTCH_Num * const    edgetab,              /* Edge array [edgenbr]                */
      &edlotab[0]); // const SCOTCH_Num * const    edlotab)              /* Edge load array                     */
    //assert (SCOTCH_graphCheck (&loggdat) == 0);
  }
  SCOTCH_stratInit (&logsdat);
  SCOTCH_stratGraphMapBuild (&logsdat, SCOTCH_STRATBALANCE, mapping_nodes.size (), 0.05);
  SCOTCH_graphMap (&loggdat, &phyadat, &logsdat, &mapping[0]); // Compute mapping of logical graph to physical graph (mapping of SCOTCH_Num kind)
  SCOTCH_stratExit (&logsdat);
  SCOTCH_graphExit (&loggdat);                    // Free logical graph data
  SCOTCH_archExit  (&phyadat);                    // Free logical graph to physical graph static mapping strategy
#else
  for(int i=0; i<mapping.size(); ++i) mapping[i] = i; // return identity mapping 
  if(!r) printf("******* SCOTCH mapping requested but not compiled in, returning identity mapping! ********\n");
#endif

  return 0;
}

/* greedily maps a logical topology graph (ltg) to a physical topology
 * graph (ptg) with nprocs[i] processors per vertex i. Results in a
 * mapping from ltg to ptg in mapping. The greedy algorithm begins at
 * the vertex start */
static int TPM_Map_greedy(int r, TPM_Graph *ptg_ref, TPM_Graph_ewgts *ptgc_ref, TPM_Graph_vwgts *nprocs_ref, TPM_Graph *ltg_ref, TPM_Graph_ewgts *weights_ref, TPM_Mapping *mapping_ref, int start) {
  // turn refs into normal objects
  TPM_Graph& ptg = *ptg_ref;
  TPM_Graph& ptgc = *ptgc_ref; // ptg edge capacities!
  TPM_Graph& ltg = *ltg_ref;
  TPM_Graph& ltgw = *weights_ref; // ltg edge weights!
  TPM_Graph_vwgts& nprocs = *nprocs_ref;
  TPM_Mapping& mapping = *mapping_ref;

  // number of slots in ptg (nprocs) must be identical to number of
  // nodes in ltg! This is just a correctness check!
#ifdef CHECKS
  if(std::accumulate(nprocs.begin(), nprocs.end(), 0) != ltg.size()) { 
    if(!r) printf("number of slots does not match (nprocs: %i, ltg.size(): %i!\n", std::accumulate(nprocs.begin(), nprocs.end(), 0), ltg.size());
    MPI_Abort(MPI_COMM_WORLD, 5);
  }
#endif

  TPM_Graph_dewgts w(ptg.size());
  // init edge weights in ptg to max(nprocs)*n^2 to force shortest paths!
  int maxw=0; 
  for(int i=0; i<ltg.size(); ++i) maxw=std::max(maxw, *std::max_element(ltgw[i].begin(), ltgw[i].end()) );
  double init = (*std::max_element(nprocs.begin(), nprocs.end())) * maxw * ptg.size()*ptg.size();
  assert(init > 0); // check for overflow
  for(int i=0; i<ptg.size(); ++i) w[i].resize(ptg[i].size(), init);

  TPM_Graph_dists ptdists(ptg.size());
  TPM_Graph_preds ptpreds(ptg.size());

  
  std::vector<bool> mapped(ltg.size(), false);
  std::vector<bool> discovered(ltg.size(), false);

  /********************************************************
   * The actual mapping code                            
   ********************************************************/

  int current=start; // the current vertex in ptg to be mapped to
  int num_map=0; // the number of mapped vertices in ltg
  while(num_map < ltg.size()) {
    /* find vertex in ltg with most edges (most edge weights) that is
     * not mapped yet and map it to next free vertex in ptg (starting
     * with myself ) */
    // TODO: linear for now, should be done with sort ...
    int maxedges=-1;
    int maxvert=-1;
    for(int i=0; i<ltg.size(); ++i) {
      if(mapped[i]) continue;
      int thisedges=0;
      for(int j=0; j<ltg[i].size(); ++j) {
        assert(i<ltgw.size() && j<ltgw[i].size());
        thisedges+=ltgw[i][j]; // edge weight of logical edge
      }
      //printf("%i -- %i\n", i, ltg[i].size());
      if(thisedges > maxedges) {
        maxedges=thisedges;
        maxvert=i;
      }
    }

    DBG4(if(!r) printf("most 'expensive' vertex in logical topology graph: %i (cost: %i)\n", maxvert, maxedges));

    // if current vertex in ptg does not have free slots, choose different
    // (pretty much random) one -- should this pick a particular one?
    while(nprocs[current] <= 0) { current = (current+1)%nprocs.size(); }

    DBG4(if(!r) printf("mapped %i to %i\n", maxvert, current));
    discovered[maxvert] = true;
    mapped[maxvert] = true;
    num_map++;
    mapping[maxvert] = current; // map rank maxvert to current
    nprocs[current]--; // one of the procs of current is occupied now
    /* add all edges of this vertex that lead to not-discovered vertices
     * to prio queue */
    std::priority_queue<std::pair<double,int>, std::vector<std::pair<double,int> >, max_compare_func > Q;
    for(int j=0; j<ltg[maxvert].size(); ++j) {
      if(!discovered[ltg[maxvert][j]]) {
        Q.push(std::make_pair(ltgw[maxvert][j], ltg[maxvert][j]));
        discovered[ltg[maxvert][j]] = 1;
      }
    }
    
    /* while Q not empty find most expensive edge in queue */
    while(!Q.empty()) {
      // take the most expensive edge out of the queue
      std::pair<double,int> Qel = Q.top();
      Q.pop();
      int target = Qel.second;

      DBG4(if(!r) printf("most expensive edge in topology graph leads to: %i (cost: %i)\n", target, Qel.first));
      
      /* find next vertex that is as close as possible (minimum path from
       * current but still has available slots (vertex weight is not 0)
       * map the target of the edge from previous step to it */
      TPM_SSSP(&ptg, &w, current, &ptdists, &ptpreds);
      double smallestdist=std::numeric_limits<double>::max();
      int smallestdistvert=-1;
      // again linear time, should be logarithmic!
      for(int j=0; j<ptdists.size(); ++j) if(smallestdist > ptdists[j] && nprocs[j] > 0) {
        smallestdist = ptdists[j];
        smallestdistvert = j;
      }
      DBG4(if(!r) printf("vertex with smallest distance from current (%i) is %i (dist %.2f)\n", current, smallestdistvert, smallestdist/ptg.size()/ptg.size()));

      // map most expensive process to the nearest physical neighbor!
      DBG4(if(!r) printf("mapped %i to %i\n", target, smallestdistvert));
      nprocs[smallestdistvert]--;
      mapping[target] = smallestdistvert;      
      mapped[target] = true;
      num_map++;

      // update occupied edges in graph
      int v=smallestdistvert;
      while(v != current) {
        // find position of v in weights map (same as in h)
        int pos; for(pos=0; ptg[ptpreds[v]][pos] != v; pos++);
        assert(w.size() > ptpreds[v]);
        assert(w[ptpreds[v]].size() > pos);
        w[ptpreds[v]][pos] += Qel.first/ptgc[ptpreds[v]][pos];
        //printf("increasing edge weight on edge %i -> %i to %i\n", pred[v], v, w[pred[v]][pos]-nh*nh);
        v = ptpreds[v];
      }

      /* add all edges of just mapped vertex (target) that lead to
       * not-discovered vertices to prio queue */
      for(int j=0; j<ltg[target].size(); ++j) {
        if(!discovered[ltg[target][j]]) {
          Q.push(std::make_pair(ltgw[target][j], ltg[target][j]));
          discovered[ltg[target][j]] = 1;
        }
      }
    }
  }
  return 0;
}

static int TPM_Map_annealing(int r, TPM_Graph *ptg_ref, TPM_Graph_ewgts *ptgc_ref, TPM_Graph_vwgts *nprocs_ref, TPM_Graph *ltg_ref, TPM_Graph_ewgts *weights_ref, TPM_Mapping *mapping_ref, double *cong, int *dil, int *iters) {
  TPM_Graph& ptg = *ptg_ref;
  TPM_Graph_ewgts& ptgc = *ptgc_ref;
  TPM_Graph& ltg = *ltg_ref;
  TPM_Graph_ewgts& ltgw = *weights_ref; // ltg edge weights!
  //TPM_Graph_vwgts& nprocs = *nprocs_ref;
  TPM_Mapping& mapping = *mapping_ref;
  MTRand mtrand;

  /* this uses threshold accepting (TA) which might be superior to
   * simulated annealing -- see Dueck and Scheuer: "Threshold accepting:
   * a general purpose optimization algorithm appearing superior to
   * simulated annealing" (Journal of Computational Physics,
   * 90(1):161-175, 1990) */
  // choose threshold value T
  int T=100;
  // threshold factor (0<TF<1) -- normalized to 100
  int TF=10; 
  // max tests 
  int Zmax=30;
  // start config == mapping
  TPM_Mapping C(mapping.size()); std::copy(mapping.begin(), mapping.end(), C.begin());
  // reference solution
  TPM_Mapping Cref(mapping.size()); std::copy(mapping.begin(), mapping.end(), Cref.begin());
  // cost of reference
  double Cref_cost;
  int Cref_dil;
  TPM_Get_cong(r, &ptg, &ptgc, &ltg, &ltgw, &Cref, &Cref_cost, &Cref_dil);

  // max. test time in seconds
  int stop=10;
  int ctr=0;

  double timeout = MPI_Wtime()+stop;
  while(MPI_Wtime() < timeout || ctr < 2) {
    for(int z=0; z<Zmax; z++) {
      // choose new config -- exchange two positions randomly
      TPM_Mapping Ctmp(mapping.size()); std::copy(mapping.begin(), mapping.end(), Ctmp.begin());
      int src = mtrand.randInt(mapping.size()-1);
      int tgt = mtrand.randInt(mapping.size()-1);
      int x=Ctmp[src];
      Ctmp[src] = Ctmp[tgt];
      Ctmp[tgt] = x;
      // get costs of new config
      double cost;
      int dil;
      TPM_Get_cong(r, &ptg, &ptgc, &ltg, &ltgw, &Ctmp, &cost, &dil);
      if(cost < Cref_cost) {
        std::copy(Ctmp.begin(), Ctmp.end(), Cref.begin());
        Cref_cost = cost; 
        Cref_dil = dil;
      }
      if(cost - std::min(cost, Cref_cost) > -T) std::copy(Ctmp.begin(), Ctmp.end(), C.begin());
    }
    T = T*TF/100;
    ctr++;
  }
  //INFO(if(!r) printf("[annealing: %i iterations took %f s]\n", ctr, MPI_Wtime()-timeout+stop));
  *iters = ctr;
  std::copy(Cref.begin(), Cref.end(), mapping.begin());
  *cong = Cref_cost;
  *dil = Cref_dil;
  return 0;
}

static inline int TPM_Graph_size(TPM_Graph *g_ref) {
  TPM_Graph& g = *g_ref;
  int ret=0;
  for(int i=0; i<g.size(); ++i) ret += g[i].size();
  return ret;
}

int TPM_Topomap(MPI_Comm distgr, const char *topofile, int max_mem, int *newrank) {
  int r,p;
  MPI_Comm_size(distgr, &p);
  MPI_Comm_rank(distgr, &r);

  TPM_Nodenames names;
  TPM_Graph  ptg; // physical topology graph 
  TPM_Graph_ewgts  ptgc; // physical topology graph capacity

#ifdef BGP
  int me;
  TPM_Get_bgp_topo(r, &ptg, &me);

  //  fill ptgc with 1
  ptgc.resize(ptg.size());
  for(int i=0; i<ptgc.size(); ++i) {
    ptgc[i].resize(ptg[i].size());
    std::fill(ptgc[i].begin(), ptgc[i].end(), 1);
  }
#else
  TPM_Read_topo(topofile, &names, &ptg, &ptgc);
  int me = TPM_Get_myid(r, &names);
#endif

  int n=ptg.size();

  const int NONE = 0; // no mapper
  const int RCM = 1; // RCM mapper
  const int GREEDY = 2; // greedy mapper
  const int RECURSIVE = 3; // recursive mapper
  const int SCOTCH = 4; // recursive mapper

  int mapper = NONE; // which mapping algorithm to use, can be different on different processes!

  int use_anneal = 1; // use simulated annealing to refine result
  if(n>500) use_anneal = 0; // too slow in those cases :-(
  int use_parmetis = 1; // variable that indicates if parmetis can be used or not (will be disabled if not all nodes have the same number of processes OR ltg is not symmetric!)
  int fix_parmetis = 1; // attempt to fix parmetis partition if it is not perfectly balanced. The fix just balances it randomly and guarantees bad outcome :-/
  int fix_graph= 1; // this is a DIRTY hack if the graph is not symmetric (ParMETIS needs a symmetric graph), make it symmetric by adding edges!

  double t_parm=0, t_map=0;

  /* check if environment variables determine our selection! -- mainly
   * for testing purposes */
  char *env = getenv("TPM_STRATEGY");
  if(env != NULL) if(!strcmp(env,"none")) mapper=NONE;
                  else if(!strcmp(env,"greedy")) mapper=GREEDY;
                  else if(!strcmp(env,"recursive")) mapper=RECURSIVE;
                  else if(!strcmp(env,"rcm")) mapper=RCM;
                  else if(!strcmp(env,"scotch")) mapper=SCOTCH;
  env = getenv("TPM_ANNEAL");
  if(env != NULL ) if(!strcmp(env,"yes")) use_anneal=1; else use_anneal=0;
  env = getenv("TPM_PARMETIS");
  if(env != NULL) if(!strcmp(env,"yes")) use_parmetis=1; else use_parmetis=0;
  env = getenv("TPM_FIX_GRAPH");
  if(env != NULL) if(!strcmp(env,"yes")) fix_graph=1; else fix_graph=0;
  env = getenv("TPM_FIX_PARMETIS");
  if(env != NULL) if(!strcmp(env,"yes")) fix_parmetis=1; else fix_parmetis=0;

  // RCM mapper needs ParMETIS!
  if(mapper == RCM) fix_parmetis = fix_graph = 1;

  const char *fake_file="none"; if(TPM_Fake_names_file != NULL) fake_file=TPM_Fake_names_file;
  INFO(if(!r) printf("INFO p: %i, fake_file: \"%s\", topo_file: \"%s\", mapper: %i (0=none, 1=rcm, 2=greedy, 3=recursive, 4=scotch), use_anneal: %i, use_parmetis: %i, fix_parmetis: %i, fix_graph: %i\n", p, fake_file, topofile, mapper, use_anneal, use_parmetis, fix_parmetis, fix_graph));

  // determine the number of processes that run on each host
  // all processes need to know the result -- this would be a
  // Gather_reduce :-) -- for now an Alltoall though :-/
  TPM_Graph_vwgts nprocs(ptg.size());
  std::fill(nprocs.begin(), nprocs.end(), 0);
  // alltoall buffers --- TWO!!! I WANT MPI-2.2!!!
  {
    std::vector<int> sbuf(p), rbuf(p);
    std::fill(sbuf.begin(), sbuf.end(), me);
    MPI_Alltoall(&sbuf[0], 1, MPI_INT, &rbuf[0], 1, MPI_INT, distgr);
    // fill nprocs structure for each vertex
    for(int i=0; i<p; ++i) nprocs[rbuf[i]]++;
  }
  
  DBG2(if(!r) for(int i=0; i<n; ++i) if(nprocs[i]) printf("%i ranks on node %i (%s)\n", nprocs[i], i, names[i].c_str()));
  
  /* the hardware graph is initialized at this point -- the weight of
   * each vertex is the number of processes that run on this vertex */
  MEM(if(!r) printf("ptg: cardinality %i, size: %i (mem: %.2f kiB)\n", ptg.size(), TPM_Graph_size(&ptg), (float)TPM_Graph_size(&ptg)*(float)sizeof(int)/1024));

  // check if all nodes have the same number of processes! This is
  // required for the metis partitioning!
  int nproc=0;
  for(int i=0; i<ptg.size(); ++i) if(nprocs[i] != 0) {
    if(nproc==0) {
      nproc = nprocs[i];
    }
    if(nproc != nprocs[i]) {
      nproc = -1;
      break;
    }
  }
  
  if(use_parmetis) if(nproc < 2) { 
    INFO(if(!r) printf("no symmetric multicore allocation found (nproc=%i) disabling ParMETIS!\n", nproc));
    use_parmetis = 0;
  } DBG2(else if(!r) printf("all nodes have %i processes\n", nproc));

  // now collect the graph that we want to map to the topology ... we
  // assume that each process *can* store the whole graph! This needs to
  // be checked before collecting it (we just assume it here). 
  
  // now, each process queries its neighbors (all edges that it has) and
  // then all those edges are allgatherv'd
#ifndef FAKE_MPI22
  int topo;
  MPI_Topo_test(distgr, &topo);
  if(topo != MPI_DIST_GRAPH) {
    printf("[%i] this is not a distributed graph communicator! Aborting job.\n", r);
    MPI_Abort(distgr, 6);
  }
#endif

  // get information about local neighbors (we need only out-vertices
  // ... we could save memory here, but I don't want to trigger bugs yet ;)
  int indegree, outdegree, weighted;
  MPIX_Dist_graph_neighbors_count(distgr, &indegree, &outdegree, &weighted);
  //assert(!weighted); // the code below works only for unweighted graphs (but accumulates multiple edges into weights)

  std::vector<int> in(indegree), out(outdegree), inw(indegree), outw(outdegree);
  MPIX_Dist_graph_neighbors(distgr, indegree, &in[0], &inw[0], outdegree, &out[0], &outw[0]);
  assert(outdegree == out.size());

  // we might get double edges here -- transform them into a weight map
  // (TODO: we loose the actual edge weights in this process but they
  // can easily be recovered if needed!)

  std::vector<int> wgts;

  // unweighted graphs might indicate weight with multiple edges
  if(!weighted) {
    if(outdegree > 0) {
      std::vector<int> tmp(outdegree); // temp vector

      std::sort(out.begin(), out.end());
      std::copy(out.begin(), out.end(), tmp.begin());
      out.erase(unique(out.begin(), out.end()), out.end());
      outdegree = out.size();
        
      // get weights by counting elements in tmp
      wgts.resize(outdegree);

      int elem=tmp[0], cnt=0, pos=0;
      for(int i=0; i<tmp.size()+1; ++i) {
        if(i < tmp.size() && tmp[i] == elem) cnt++;
        else { 
          if(i<tmp.size()) elem = tmp[i]; // this if guard seems silly -- might want to restructure this whole loop!
          assert(pos < out.size());
          //DBG2(printf("[%i] weight[%i]=%i\n", r, pos, cnt));
          wgts[pos++] = cnt;
          cnt=1; 
  } } } } else {
    // weigthed graphs ignore double edges TODO: should be merged too at
    // some point ...
    wgts.resize(outdegree);
    std::copy(outw.begin(), outw.end(), wgts.begin());
  }



  //printf("[%i] in: %i, out %i\n", r, indegree, outdegree);

  TPM_Graph ltg(p);
  TPM_Graph_ewgts ltgw(ltg.size());
  // now collect information for allgatherv -- we only send out-vertices!
  {
    std::vector<int> rcounts(p);
    MPI_Allgather(&outdegree, 1, MPI_INT, &rcounts[0], 1, MPI_INT, distgr);
    int bufsize = std::accumulate(rcounts.begin(), rcounts.end(), 0);
    
    std::vector<int> rbuf(bufsize);
    std::vector<int> displs(p);
    displs[0] = 0; for(int i=1; i<p; ++i) displs[i] = displs[i-1]+rcounts[i-1]; 
    MPI_Allgatherv(&out[0], outdegree, MPI_INT, &rbuf[0], &rcounts[0], &displs[0], MPI_INT, distgr);

    // weights
    std::vector<int> wgtsrbuf(bufsize);
    MPI_Allgatherv(&wgts[0], outdegree, MPI_INT, &wgtsrbuf[0], &rcounts[0], &displs[0], MPI_INT, distgr);

    // save full graph in ltg
    for(int i=0; i<p; ++i) {
      DBG2(if(r==0) printf("vertex %i adjacency list: ", i));
      ltg[i].resize(rcounts[i]);
      ltgw[i].resize(rcounts[i]);
      for(int j=displs[i]; j<displs[i]+rcounts[i]; j++) {
        ltg[i][j-displs[i]] = rbuf[j];
        ltgw[i][j-displs[i]] = wgtsrbuf[j];
        DBG2(if(r==0) printf(" %i (%i) ", rbuf[j], wgtsrbuf[j]));
      }
      DBG2(if(r==0) printf("\n"));
    }
  }

  /* at this point each process has access to the physical topology
   * graph ptg and the logical topology graph ltg */
  MEM(if(!r) printf("ltg: cardinality %i, size: %i (mem: %.2f kiB)\n", ltg.size(), TPM_Graph_size(&ltg), (float)TPM_Graph_size(&ltg)*(float)sizeof(int)/1024));

  /* get the edge cut with the default mapping */
  TPM_Mapping origmapping(p);
  double origmax_cong;
  int origmax_dil;
  double origavgs[2];
  // gather the mapping from all processes - \Theta(P) space
  MPI_Allgather(&me, 1, MPI_INT, &origmapping[0], 1, MPI_INT, distgr);
  TPM_Get_cong(r, &ptg, &ptgc, &ltg, &ltgw, &origmapping, &origmax_cong, &origmax_dil, false, &origavgs[0], &origavgs[1], "origcong.dot");
  //if(!r) printf("orig max. congestion: %i, max. dilation %i\n", origmax_cong, origmax_dil);
  /* end getting edge cut for default mapping */


  // the ltg graph *must* be symmetric for METIS!
  // check ltg for symmetry
  if(use_parmetis) for(int i=0; i<ltg.size(); ++i) {
    for(int j=0; j<ltg[i].size(); ++j) {
      int src = i, tgt = ltg[i][j], found=0;
      //if(!r) printf("%i -> %i (%i/%i)\n", src, tgt, j, ltg[i].size());
      for(int k=0; k<ltg[tgt].size(); ++k) if(ltg[tgt][k] == src) found=1;
      if(!found) if(fix_graph) {
        // add edge that does not exist!
        ltg[tgt].push_back(src);
        ltgw[tgt].push_back(ltgw[src][tgt]);
        INFO(if(!r) printf("added edge %i->%i to make graph symmetric\n", tgt, src));
      } else {
        INFO(if(!r) printf("edge (%i,%i) exists but edge (%i,%i) not! Graph not symmetric - disabling ParMETIS\n", src,tgt,tgt,src));
        use_parmetis = 0;
        break;
      }
    } 
    if(!use_parmetis) break;
  }

  TPM_Mapping mapping(ltg.size()); // topology mapping
  std::vector<int> fullpart(p); // full ParMETIS partition vector
  
  if(use_parmetis) {
    assert(ltg.size() == p);
    std::vector<idxtype>  vtxdist(p+1), // vertex distribution
                          xadj(2), // CSR index 
                          adjncy(ltg[r].size()); // CSR list

    vtxdist[0] = 0; for(int i=1; i<(p+1); i++) vtxdist[i]=vtxdist[i-1]+1;
    xadj[0]=0; xadj[1]=ltg[r].size();
    
    std::copy(ltg[r].begin(), ltg[r].end(), adjncy.begin());

    // make sure that the initializer is bigger than all edge weights
    // (needed to guarantee that parmetis balances strictly!!!!
    idxtype initializer=0; for(int i=0; i<ltgw.size(); ++i) initializer += std::accumulate(ltgw[i].begin(), ltgw[i].end(), 0); 
    std::vector<idxtype> vwgt(1,initializer); // vertex weights 
    std::vector<idxtype> adjwgt(ltg[r].size()); // edge weights
    std::copy(ltgw[r].begin(), ltgw[r].end(), adjwgt.begin());

    //for(int i=0; i<adjncy.size(); i++) printf("[%i] -> %i (%i)\n", r, adjncy[i], adjwgt[i]);
    int wgtflag=3; // 0 No weights, 1 Weights on the edges only, 2 Weights on the vertices only, 3 Weights on both the vertices and edges. 
    int numflag=0; // numbering scheme - 0 C-style numbering that starts from 0
    int ncon=1; // number of weights that each vertex has.
    int nparts = p/nproc; // number of partitions
    std::vector<float> tpwgts(ncon*nparts,1.0/nparts); // specify the fraction of vertex weight 
    std::vector<float> ubvec(ncon,1.05); // specify the imbalance tolerance for each vertex weight
    int options[3] = {0,0,0};
    int edgecut;
    std::vector<idxtype> part(1);

    if(nparts != p) {
      MPI_Barrier(distgr); // for timing only -- I know, it's dumb
      t_parm = -MPI_Wtime();
      ParMETIS_V3_PartKway(&vtxdist[0], &xadj[0], &adjncy[0], &vwgt[0], &adjwgt[0], &wgtflag, &numflag, 
                        &ncon, &nparts, &tpwgts[0], &ubvec[0], options, &edgecut, &part[0], &distgr);
      MPI_Barrier(distgr); // for timing only -- I know, it's dumb
      t_parm += MPI_Wtime();
    } else {
      part[0] = r;
      edgecut = -1; 
    }
    INFO(if(!r) printf("[ParMETIS: edge-cut: %i (splitted graph of cardinality %i in %i partitions, nproc=%i)]\n", edgecut, ltg.size(), nparts, nproc));

    /* build node-graph from the parmetis partition */
    // first collect the parmetis partition everywhere
    assert(sizeof(idxtype) == sizeof(int));
    fullpart.resize(p);
    MPI_Allgather(&part[0], 1, MPI_INT, &fullpart[0], 1, MPI_INT, distgr);
    DBG2(if(!r) printf("[%i] parmetis partition: ", r); if(!r) for(int i=0; i<p; ++i) printf("%i ", fullpart[i]); if(!r) printf("\n"));

    /* parmetis does not guarantee to generate perfectly balanced
     * partitions. They are most likely well balanced but our
     * application *requires* perfect balancing. Thus, we might need to
     * correct the partitioning here */
    std::vector<int> cnts(p,0);
    // traverse fullpart and check if each host exists exactly nprocs times!
    for(int i=0; i<p; ++i) cnts[fullpart[i]]++;
    int balanced = 1;
    for(int i=0; i<nparts; ++i) if(cnts[i] != nproc) balanced = 0;
    // fix parmetis mapping to be balanced
    if(!balanced) if(fix_parmetis) {
      int nvert_fixed=0; // number of vertices that have been fixed
      while(!balanced) {
        nvert_fixed++;
        // find overloaded node
        int overloaded = 0;
        for(int i=0; i<nparts; ++i) if(cnts[i] > nproc) { overloaded=i; break; }
        // find underloaded node 
        int underloaded = 0;
        for(int i=0; i<nparts; ++i) if(cnts[i] < nproc) { underloaded=i; break; }
        // exchange overloaded and underloaded -- TODO: we should pick
        // the least connected vertex from the overloaded node here!
        int overloaded_index=0;
        for(int i=0; i<p; ++i) if(fullpart[i] == overloaded) { overloaded_index=i; break; }
        fullpart[overloaded_index] = underloaded;
        cnts[overloaded]--;
        cnts[underloaded]++;
        
        // re-check balance
        balanced = 1; for(int i=0; i<nparts; ++i) if(cnts[i] != nproc) balanced = 0;
      }
      INFO(if(!r && nvert_fixed>0) printf("fixed (moved) %i ParMETIS vertices!\n", nvert_fixed));
    } else { 
      INFO(if(!r) printf("ParMETIS partition not perfectly balanced and fix_parmetis=false -- disabling ParMETIS\n"));
      use_parmetis = 0;
    }
  }

  if(use_parmetis) {
    // build node topology graph
    TPM_Graph ntg(p/nproc);
    TPM_Graph_ewgts ntgw(ntg.size());
    for(int i=0; i<ltg.size(); ++i) {
      for(int j=0; j<ltg[i].size(); ++j) {
        int src=i, tgt=ltg[i][j];
        int nsrc=fullpart[src];
        int ntgt=fullpart[tgt];
        // add edge to node topology graph if necessary
        if(nsrc != ntgt) {
          // check if edge exists -- TODO: change lin. search to map lookup!
          int found=0, pos=0;
          for(int k=0; k<ntg[nsrc].size(); ++k, ++pos) if(ntg[nsrc][pos] == ntgt) { found = 1; break; }
          if(found) {
            ntgw[nsrc][pos] += ltgw[i][j];
          } else {
            ntg[nsrc].push_back(ntgt);
            ntgw[nsrc].push_back(ltgw[i][j]);
          }
        }
      }
    }
    MEM(if(!r) printf("ntg: cardinality %i, size: %i (mem: %.2f kiB)\n", ntg.size(), TPM_Graph_size(&ntg), (float)TPM_Graph_size(&ntg)*(float)sizeof(int)/1024));
    
    //if(!r) for(int i=0; i<ntg.size(); ++i) { printf("ntg vertex %i adjacency list: ", i); for(int j=0; j<ntg[i].size(); j++) printf(" %i ", ntg[i][j]); printf("\n"); }

    for(int i=0; i<nprocs.size(); ++i) if(nprocs[i] > 0) nprocs[i] = 1;
    
    TPM_Mapping ntgmap(ntg.size()); // mapping from ntg to ptg

    t_map = -MPI_Wtime();
    if(mapper == RCM) TPM_Map_rcm(r, &ptg, &nprocs, &ntg, &ntgmap, me /* start */);
    else if(mapper == RECURSIVE) { // set up maps for recursive mapping
      TPM_Mapping lmap(ntg.size());
      for(int i=0; i<ntg.size(); ++i) lmap[i] = i;

      // copy ptg and ptgc graphs
      TPM_Graph tmpptg(ptg.size()), tmpptgc(ptgc.size());
      std::copy(ptg.begin(), ptg.end(), tmpptg.begin());
      for(int i=0; i<ptg.size(); i++) {
        tmpptg[i].resize(ptg[i].size());
        std::copy(ptg[i].begin(), ptg[i].end(), tmpptg[i].begin());
      }
      std::copy(ptgc.begin(), ptgc.end(), tmpptgc.begin());
      for(int i=0; i<ptgc.size(); i++) {
        tmpptgc[i].resize(ptgc[i].size());
        std::copy(ptgc[i].begin(), ptgc[i].end(), tmpptgc[i].begin());
      }
      TPM_Mapping pmap;
      for(int i=0; i<ptg.size(); ++i) pmap.push_back(i);
      
      if(!r) printf("calling recursive mapper with ltg: %i and ptg: %i\n", lmap.size(), std::accumulate(nprocs.begin(), nprocs.end(), 0));
      TPM_Map_recursive(r, &tmpptg, &tmpptgc, &nprocs, &ntg, &ntgw, &ntgmap, lmap, &pmap);
    } else if(mapper == SCOTCH) {
      TPM_Map_scotch(r, &ptg, &ptgc, &nprocs, &ntg, &ntgw, &ntgmap, me /* start */);
    } else TPM_Map_greedy(r, &ptg, &ptgc, &nprocs, &ntg, &ntgw, &ntgmap, me /* start */);
    t_map += MPI_Wtime();

    // inflate mapping back to logical topology size
    for(int i=0; i<ltg.size(); ++i) {
      int pos_in_ntg=fullpart[i];
      assert(pos_in_ntg<ntgmap.size());
      mapping[i] = ntgmap[pos_in_ntg];
    }
  } else { // if(use_parmetis)
    // do greedy mapping on full graph
    t_map = -MPI_Wtime();
    if(mapper == RCM) TPM_Map_rcm(r, &ptg, &nprocs, &ltg, &mapping, me /* start */);
    else if(mapper == RECURSIVE) { // set up maps for recursive mapping
      TPM_Mapping lmap(ltg.size());
      for(int i=0; i<ltg.size(); ++i) lmap[i] = i;
      
      // copy ptg and ptgc graphs
      TPM_Graph tmpptg(ptg.size());
      TPM_Graph_ewgts tmpptgc(ptgc.size());
      std::copy(ptg.begin(), ptg.end(), tmpptg.begin());
      for(int i=0; i<ptg.size(); i++) {
        tmpptg[i].resize(ptg[i].size());
        std::copy(ptg[i].begin(), ptg[i].end(), tmpptg[i].begin());
      }
      std::copy(ptgc.begin(), ptgc.end(), tmpptgc.begin());
      for(int i=0; i<ptgc.size(); i++) {
        tmpptgc[i].resize(ptgc[i].size());
        std::copy(ptgc[i].begin(), ptgc[i].end(), tmpptgc[i].begin());
      }
      TPM_Mapping pmap;
      for(int i=0; i<ptg.size(); ++i) pmap.push_back(i);
      
      if(!r) printf("calling recursive mapper with ltg: %i and ptg: %i\n", lmap.size(), std::accumulate(nprocs.begin(), nprocs.end(), 0));
      TPM_Map_recursive(r, &tmpptg, &tmpptgc, &nprocs, &ltg, &ltgw, &mapping, lmap, &pmap);
    } else if(mapper == SCOTCH) {
      TPM_Map_scotch(r, &ptg, &ptgc, &nprocs, &ltg, &ltgw, &mapping, me /* start */);
    } else TPM_Map_greedy(r, &ptg, &ptgc, &nprocs, &ltg, &ltgw, &mapping, me /* start */);
    t_map += MPI_Wtime();
  } // else(use_parmetis)

  double max_cong;
  int max_dil;
  double avgs_gr[3];
  TPM_Get_cong(r, &ptg, &ptgc, &ltg, &ltgw, &mapping, &max_cong, &max_dil, false, &avgs_gr[1], &avgs_gr[2]);
  avgs_gr[0] = max_dil;
  //if(!r) printf("max. congestion: %i, max. dilation %i\n", max_cong, max_dil);

  // only to print it later ... 
  doubleint ti_greedy = { max_cong, r };
  MPI_Allreduce(MPI_IN_PLACE, &ti_greedy, 1, MPI_DOUBLE_INT, MPI_MINLOC, distgr);
  MPI_Bcast(avgs_gr, 3, MPI_DOUBLE, ti_greedy.b, distgr);

  double new_cong;
  int new_dil, anneal_iters=0;
  double t_anneal = -MPI_Wtime();
  if(use_anneal) {
    // bcast best mapping such that annealing step starts with best! --
    // this makes the results WORSE (against all expectations) -- at least
    // for our test-cases ... thus disable for now
    /* each process computed a different mapping here -- see which one
     * found the best one! */
    /*MPI_Bcast(&mapping[0], p, MPI_INT, ti_greedy.b, distgr);*/
    
    /* do optimization */
    TPM_Map_annealing(r, &ptg, &ptgc, &nprocs, &ltg, &ltgw, &mapping, &new_cong, &new_dil, &anneal_iters);
    //printf("[%i] annealing %i -> %i (dil: %i -> %i)\n", r, max_cong, new_cong, max_dil, new_dil);
  } else {
    INFO(if(!r) printf("annealing disabled\n"));
    new_cong = max_cong;
    new_dil = max_dil;
  }
  t_anneal += MPI_Wtime();

  if(!r) printf("ParMETIS time: %.6f s; Mapping time strategy %i: %.6f s; Annealing time: %.6f iters: %i\n", t_parm, t_map, mapper, t_anneal, anneal_iters);

  double avgs[2];
  TPM_Get_cong(r, &ptg, &ptgc, &ltg, &ltgw, &mapping, &new_cong, &new_dil, false, &avgs[0], &avgs[1], "cong.dot");
  doubleint ti = { new_cong, r };
  MPI_Allreduce(MPI_IN_PLACE, &ti, 1, MPI_DOUBLE_INT, MPI_MINLOC, distgr);
  MPI_Bcast(avgs, 2, MPI_DOUBLE, ti.b, distgr);

  int new_mapping_found=1;
  if(ti.a >= origmax_cong) {
    max_cong=origmax_cong;
    max_dil=origmax_dil;
    INFO(if(!r) printf("no better mapping found with strategy %i (found cong: %.2f, %i) -- using orig (cong: %.2f, %i (%.2f, %.2f))\n", mapper, new_cong, new_dil, max_cong, max_dil, origavgs[0], origavgs[1]));
    *newrank = r;
    new_mapping_found=0;
    mapping = origmapping;
  } else {
    double data[3]={new_dil, avgs[0], avgs[1]};
    if(ti.b != 0) {
      /* get dilation to rank 0 -- only for printing */
      MPI_Request req;
      if(!r) MPI_Irecv(&data, 3, MPI_DOUBLE, ti.b, 0, distgr, &req);
      if(r==ti.b) MPI_Send(&data, 3, MPI_DOUBLE, 0, 0, distgr);
      if(!r) MPI_Wait(&req, MPI_STATUS_IGNORE);
      /* end get dilation */
    }

    INFO(if(!r) printf("process %i found minimum congestion %.2f, %.0f (%.2f, %.2f) (orig: %.2f, %i (%.2f, %.2f); strategy %i: %.2f, %.0f (%.2f, %.2f))\n", ti.b, ti.a, data[0], data[1], data[2], origmax_cong, origmax_dil, origavgs[0], origavgs[1], mapper, ti_greedy.a, avgs_gr[0], avgs_gr[1], avgs_gr[2]));
    // broadcast best mapping
    MPI_Bcast(&mapping[0], p, MPI_INT, ti.b, distgr);
  }
  
   
  // I am rank r and run on node me, mapping[r] indicates the new node
  // where I want to go (newme). 
  int newme = mapping[r]; // my target node
#ifdef HAVE_NODE_MAPPING 
  // if on-node mapping is provided, call 
  std::vector<int> neighbors; // ranks in ltg that are on the same node as I am
  TPM_Graph ntg; // on-node topology graph -- ntg[i] contains vertex neighbor[i]
  TPM_Graph_ewgts ntgw; // weights of on-node graph
  std::map<int,int> neighmap; // neighbor map for faster lookup
  
  // print mapping
  //if(!r) for(int i=0; i<p; ++i) printf("%i ", mapping[i]); if(!r) printf("\n");
  
  for(int i=0; i<p; ++i) if(mapping[i] == newme) { neighmap[i]=neighbors.size(); neighbors.push_back(i); }
  ntg.resize(neighbors.size());
  ntgw.resize(neighbors.size());

  // extract subgraph of ltg that connects neighbors
  for(int i=0; i<neighbors.size(); ++i) {
    int u = neighbors[i]; // source vertex
    for(int j=0; j<ltg[u].size(); ++j) {
      int v = ltg[u][j]; // target vertex
      std::map<int,int>::iterator n = neighmap.find(v);
      if (n != neighmap.end()) {
        ntg[i].push_back((*n).second);
        ntgw[i].push_back(ltgw[u][j]);
      }
    }
  }

  *newrank = TPM_Node_mapping(&ntg, &ntgw, &neighbors, distgr, r);
#else 
  if(new_mapping_found) {
    // no on-node mapping is provided, essentially random order
    int before=0; // the number of ranks smaller than r (myself) that are mapped to my target node (newme)!
    for(int i=0; i<r; ++i) if(mapping[i] == newme) before++;

    INFO(if(!r) printf("origmapping "); if(!r) for(int i=0; i<p; ++i) printf("%i ", origmapping[i]); if(!r) printf("\n"));
    INFO(if(!r) printf("mapping "); if(!r) for(int i=0; i<p; ++i) printf("%i ", mapping[i]); if(!r) printf("\n"));
    
    int pos=0;
    //printf("%i searching %i (before: %i)\n", r, newme, before);
    for(pos=0; pos<p; ++pos) {
      if(origmapping[pos] == newme) before--;  // we found a target (other ranks might be before us)
      if(before<0) break; // we found *our* target!
    }
    *newrank = pos;
  }
#endif
 

  return 0;
} // TPM_Topomap


int TPM_Write_graph_comm(MPI_Comm grphcomm, const char *filename) {
  int inneighbors, outneighbors, weighted;
  MPIX_Dist_graph_neighbors_count(grphcomm, &inneighbors, &outneighbors, &weighted);
  int r,p;
  MPI_Comm_size(grphcomm, &p);
  MPI_Comm_rank(grphcomm, &r);

  std::vector<int> sources(inneighbors), sourcesw(inneighbors), dests(outneighbors), destsw(outneighbors);
  MPIX_Dist_graph_neighbors(grphcomm, inneighbors, &sources[0], &sourcesw[0], outneighbors, &dests[0], &destsw[0]);

//if(!r) for(int i=0; i<inneighbors; ++i) printf("(%i) %i\n", sources[i], sourcesw[i]);

  for(int i=0; i<inneighbors; ++i) if(sources[i] > p || sources[i] < 0) printf("[%i] found illegal edge %i\n", r, sources[i]);

  std::vector<int> rcnts(p), rdispls(p);
  MPI_Gather(&outneighbors, 1, MPI_INT, &rcnts[0], 1, MPI_INT, 0, grphcomm);
  for(int i=0; i<p; ++i) if(i==0) rdispls[i] = 0; else rdispls[i] = rdispls[i-1] + rcnts[i-1];

  int num_edges = std::accumulate(rcnts.begin(), rcnts.end(), 0);
  std::vector<int> tmpedges(num_edges), tmpedgesw(num_edges,1);
  MPI_Gatherv(&dests[0], outneighbors, MPI_INT, &tmpedges[0], &rcnts[0], &rdispls[0], MPI_INT, 0, grphcomm);
  if(weighted) MPI_Gatherv(&destsw[0], outneighbors, MPI_INT, &tmpedgesw[0], &rcnts[0], &rdispls[0], MPI_INT, 0, grphcomm);
  
  if(r == 0) {
    std::map<std::pair<int,int>,int> alledges;
    int cnt=0;
    for(int i=0; i<p; ++i) 
      for(int j=0; j<rcnts[i]; ++j) {
        std::pair<int,int> e=std::make_pair(tmpedges[cnt], i);
        if(alledges.find(e) == alledges.end()) alledges[e] = tmpedgesw[cnt];
        else alledges[e] += tmpedgesw[cnt];
        cnt++;
      }

    FILE *f = fopen(filename, "w");
    fprintf(f, "digraph net {\n");
    for(std::map<std::pair<int,int>,int>::iterator it=alledges.begin(); it!=alledges.end(); ++it) {
      std::pair<int,int> e=it->first;
      fprintf(f, "%i -> %i [ weight=\"%i\" ]\n", e.first, e.second, it->second);
    }
    fprintf(f, "}\n");
    fclose(f);
    INFO(printf("[%s written]\n", filename));
  }
  return 0;
}

int TPM_Write_phystopo(MPI_Comm grphcomm, const char *filename, const char *topofile) {
  int r,p;
  MPI_Comm_size(grphcomm, &p);
  MPI_Comm_rank(grphcomm, &r);

  TPM_Nodenames names;
  TPM_Graph  ptg; // physical topology graph 
  TPM_Graph_ewgts  ptgw; // physical topology graph weights
  TPM_Read_topo(topofile, &names, &ptg, &ptgw);

  int n=ptg.size();

  int me = TPM_Get_myid(r, &names);

  // determine the number of processes that run on each host
  // all processes need to know the result -- this would be a
  // Gather_reduce :-) -- for now an Alltoall though :-/
  TPM_Graph_vwgts nprocs(ptg.size());
  std::fill(nprocs.begin(), nprocs.end(), 0);
  // alltoall buffers --- TWO!!! I WANT MPI-2.2!!!
  std::vector<int> sbuf(p), rbuf(p);
  std::fill(sbuf.begin(), sbuf.end(), me);
  MPI_Alltoall(&sbuf[0], 1, MPI_INT, &rbuf[0], 1, MPI_INT, grphcomm);
  // fill nprocs structure for each vertex
  for(int i=0; i<p; ++i) nprocs[rbuf[i]]++;
  
  
  DBG2(if(!r) for(int i=0; i<n; ++i) if(nprocs[i]) printf("%i ranks on node %i (%s)\n", nprocs[i], i, names[i].c_str()));
  
  /* the hardware graph is initialized at this point -- the weight of
   * each vertex is the number of processes that run on this vertex */
  
  if(r == 0) {
    FILE *f = fopen(filename, "w");
    fprintf(f, "digraph net {\n");
    // print rank allocations!
    for(int i=0; i<ptg.size(); ++i) if(nprocs[i] > 0) {
      fprintf(f, "%i [ ranks=\"", i);
      int first=1;
      for(int j=0; j<p; ++j) if(rbuf[j] == i) {
        if(first) { first=0; } else { fprintf(f, ","); }
        fprintf(f, "%i", j);
      }
      fprintf(f, "\" ]\n");
    }
    
    for(int i=0; i<ptg.size(); ++i) 
      for(int j=0; j<ptg[i].size(); ++j)
        fprintf(f, "%i -> %i \n", i, ptg[i][j]);
    fprintf(f, "}\n");
    fclose(f);
    INFO(printf("[%s written]\n", filename));
  }

  return 0;
}

void TPM_Benchmark_graphtopo(MPI_Comm distgr, int newrank, int dsize, double *before, double *after) {
  int r,p;
  MPI_Comm_size(distgr, &p);
  MPI_Comm_rank(distgr, &r);

  int indegree, outdegree, weighted;
  MPIX_Dist_graph_neighbors_count(distgr, &indegree, &outdegree, &weighted);
  //assert(!weighted); // the code below works only for unweighted graphs (but accumulates multiple edges into weights)

  std::vector<int> in(indegree), out(outdegree), inw(indegree), outw(outdegree);
  MPIX_Dist_graph_neighbors(distgr, indegree, &in[0], &inw[0], outdegree, &out[0], &outw[0]);

  //-- stolen from TPM_Topomap() !!!
  // we might get double edges here -- transform them into a weight map
  // (TODO: we loose the actual edge weights in this process but they
  // can easily be recovered if needed!)
  //std::vector<int> outw;
  if(!weighted) {
    if(outdegree > 0) {
      std::vector<int> tmp(out.size()); // temp vector

      std::sort(out.begin(), out.end());
      std::copy(out.begin(), out.end(), tmp.begin());
      out.erase(unique(out.begin(), out.end()), out.end());
      outdegree = out.size();
        
      // get weights by counting elements in tmp
      outw.resize(outdegree);

      int elem=tmp[0], cnt=0, pos=0;
      for(int i=0; i<tmp.size()+1; ++i) {
        if(tmp[i] == elem && i<tmp.size()) cnt++;
        else { 
          elem = tmp[i]; 
          assert(pos < out.size());
          //DBG2(printf("[%i] weight[%i]=%i\n", r, pos, cnt));
          outw[pos++] = cnt;
          cnt=1; 
  } } } } else {
    // weigthed graphs ignore double edges TODO: should be merged too at
    // some point ...
    assert(outw.size() == outdegree);
  }

  //std::vector<int> inw;
  if(!weighted) {
    if(indegree > 0) {
      std::vector<int> tmp(in.size()); // temp vector

      std::sort(in.begin(), in.end());
      std::copy(in.begin(), in.end(), tmp.begin());
      in.erase(unique(in.begin(), in.end()), in.end());
      indegree = in.size();
        
      // get weights by counting elements in tmp
      inw.resize(indegree);

      int elem=tmp[0], cnt=0, pos=0;
      for(int i=0; i<tmp.size()+1; ++i) {
        if(tmp[i] == elem && i<tmp.size()) cnt++;
        else { 
          elem = tmp[i]; 
          assert(pos < in.size());
          //DBG2(printf("[%i] weight[%i]=%i\n", r, pos, cnt));
          inw[pos++] = cnt;
          cnt=1; 
  } } } } else {
    // weigthed graphs ignore double edges TODO: should be merged too at
    // some point ...
    assert(inw.size() == indegree);
  }

  /* do benchmark - send along edges */

  static const int mult=dsize;
  static const int trials=11;
  std::vector<std::vector<char> > sbufs(outdegree), rbufs(indegree);
  for(int i=0; i<outdegree; ++i) sbufs[i].resize(mult*outw[i]);
  for(int i=0; i<indegree; ++i) rbufs[i].resize(mult*inw[i]);
  std::vector<MPI_Request> sreqs(outdegree), rreqs(indegree);

  DBG5(if(!r) printf("starting benchmark!\n"));

  MPI_Barrier(distgr);

  MPI_Aint w;
  double t[2];
  for(int i=0; i < trials; ++i) {
    if (i == 1) {
      t[0] = -MPI_Wtime();
    }
    for(int j=0; j<in.size(); ++j) {
      MPI_Irecv(&rbufs[j][0], inw[j]*mult, MPI_BYTE, in[j], 0, distgr, &rreqs[j]);
    }
    for(int j=0; j<out.size(); ++j) {
      MPI_Isend(&sbufs[j][0], outw[j]*mult, MPI_BYTE, out[j], 0, distgr, &sreqs[j]);
    }
    // silly AMPI doesn't support MPI_STATUSES_IGNORE
    std::vector<MPI_Status> rstats(rreqs.size());
    std::vector<MPI_Status> sstats(sreqs.size());
    MPI_Waitall(rreqs.size(), &rreqs[0], &rstats[0]);
    MPI_Waitall(sreqs.size(), &sreqs[0], &sstats[0]);
  }
  t[0] += MPI_Wtime();

  /* create reordered comm */
  MPI_Comm reorderedcomm;
  MPI_Comm_split(distgr, 0, newrank, &reorderedcomm);

  int newr;
  MPI_Comm_rank(reorderedcomm, &newr);
  assert(newr == newrank);

  DBG5(if(!r) printf("finished stage one (%f s), rearranging edge connectivity!\n", t[0]));

  /* now we need to send our neighbor list to the host that has our rank
   * now -- first figure out the permutation in a non-scalable way :) */
  std::vector<int> permutation(p);
  MPI_Allgather(&newrank, 1, MPI_INT, &permutation[0], 1, MPI_INT, distgr);
  int rpeer=permutation[r]; // new rank r in reorderedcomm!
  int speer=0; while(permutation[speer++] != r); speer--;
  //printf("[%i] >%i <%i\n", r, speer, rpeer);

  MPI_Request reqs[4];
  MPI_Isend(&in[0], in.size(), MPI_INT, speer, 1, distgr, &reqs[0]);
  assert(in.size() == inw.size());
  MPI_Isend(&inw[0], in.size(), MPI_INT, speer, 1, distgr, &reqs[1]);
  //printf("%i sending in %i %i to %i\n", r, in[0], in[1], speer);
  MPI_Isend(&out[0], out.size(), MPI_INT, speer, 2, distgr, &reqs[2]);
  assert(out.size() == outw.size());
  MPI_Isend(&outw[0], out.size(), MPI_INT, speer, 2, distgr, &reqs[3]);
  //printf("%i sending out %i %i to %i\n", r, out[0], out[1], speer);

  MPI_Status stat;
  /* tag == 1 -> in edges */
  MPI_Probe(rpeer, 1, distgr, &stat);
  int count;
  MPI_Get_count(&stat, MPI_INT, &count);
  std::vector<int> rin(count), rinw(count);
  MPI_Recv(&rin[0], count, MPI_INT, rpeer, 1, distgr, MPI_STATUS_IGNORE);
  MPI_Recv(&rinw[0], count, MPI_INT, rpeer, 1, distgr, MPI_STATUS_IGNORE);
  //printf("%i recvd in %i %i from %i\n", r, rin[0], rin[1], rpeer);

  /* tag == 2 -> out edges */
  MPI_Probe(rpeer, 2, distgr, &stat);
  MPI_Get_count(&stat, MPI_INT, &count);
  std::vector<int> rout(count), routw(count);
  MPI_Recv(&rout[0], count, MPI_INT, rpeer, 2, distgr, MPI_STATUS_IGNORE);
  MPI_Recv(&routw[0], count, MPI_INT, rpeer, 2, distgr, MPI_STATUS_IGNORE);
  //printf("%i recvd out %i %i from %i\n", r, rout[0], rout[1], rpeer);
 
  // silly AMPI doesn't support MPI_STATUSES_IGNORE
  std::vector<MPI_Status> stats(4);
  MPI_Waitall(4, reqs, &stats[0]);

  DBG5(MPI_Barrier(distgr); if(!r) printf("finished rearranging edge connectivity, starting phase two!\n"));

  /* reset all data structures for new permutation */ 
  sreqs.resize(rout.size());
  rreqs.resize(rin.size());

  sbufs.resize(rout.size());
  for(int i=0; i<sbufs.size(); ++i) sbufs[i].resize(mult*routw[i]);
  rbufs.resize(rin.size());
  for(int i=0; i<rbufs.size(); ++i) rbufs[i].resize(mult*rinw[i]);

  MPI_Barrier(distgr);

  for(int i=0; i < trials; ++i) {
    if(i == 1) {
      t[1] = -MPI_Wtime();
    }
    for(int j=0; j<rin.size(); ++j) {
      MPI_Irecv(&rbufs[j][0], rinw[j]*mult, MPI_BYTE, rin[j], 0, reorderedcomm, &rreqs[j]);
    }
    for(int j=0; j<rout.size(); ++j) {
      MPI_Isend(&sbufs[j][0], routw[j]*mult, MPI_BYTE, rout[j], 0, reorderedcomm, &sreqs[j]);
    }
    // silly AMPI doesn't support MPI_STATUSES_IGNORE
    std::vector<MPI_Status> rstats(rreqs.size());
    std::vector<MPI_Status> sstats(sreqs.size());
    MPI_Waitall(rreqs.size(), &rreqs[0], &rstats[0]);
    MPI_Waitall(sreqs.size(), &sreqs[0], &sstats[0]);
  }
  t[1] += MPI_Wtime();

  double rt[2];
  MPI_Allreduce(t, rt, 2, MPI_DOUBLE, MPI_MAX, distgr);

  *before=rt[0];
  *after=rt[1];
}

/* this function is supposed to perform on-node mapping, it is called
 * after the global communication graph has been mapped to the network
 * topology (this mapping assumed infinite in-node bandwidth). This
 * second stage gets the on-node sub-graphs as input and maps the
 * processes to cores on NUMA nodes, the inputs are:
 *   - ltg: the logical (application) graph topology as adjacency list,
 *     this is the on-node subgraph of the global graph
 *   - ltgw: the weights of the on-node graph topology (by whatever
 *     metric they're provided to MPI)
 *   - idx2rank: an array to translate vertices in ltg to ranks in the
 *     MPI communicator comm
 *   - comm: the MPI topology communicator for the mapping, can be used
 *     for MPI communication
 *   - r: the rank of the local process in comm
 * 
 * This function returns the *NEW* rank of the current process in the
 * topology mapping.
 */
int TPM_Node_mapping(TPM_Graph* ltg_ref, TPM_Graph_ewgts *ltgw_ref, TPM_Mapping* idx2rank_ref, MPI_Comm comm, int r) {
  TPM_Graph& ltg = *ltg_ref;
  TPM_Graph_ewgts& ltgw = *ltgw_ref;
  TPM_Mapping& idx2rank = *idx2rank_ref;

  printf("[%i] TPM_Node_mapping %i neighbors\n", r, ltg.size()); fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD); // this is horrible
  // print the on-node graph adjacency list 
  if(0 == r) {
    for(int i=0; i<ltg.size(); ++i) {
      printf("%i: (rank %i): ", i, idx2rank[i]);
      for(int j=0; j<ltg[i].size(); ++j) {
        printf("%i ", ltg[i][j]);
      }
      printf("\n");
    }
  }

  // identity mapping for now -- should change :-)
  return r;
}
