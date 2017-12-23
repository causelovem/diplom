/*
 * Copyright (c) 2008 The Trustees of Indiana University and Indiana
 *                    University Research and Technology
 *                    Corporation.  All rights reserved.
 *
 * Author(s): Torsten Hoefler 
 *
 */

#include <mpi.h>

extern "C" {

int MPIX_Dist_graph_create_adjacent(MPI_Comm comm_old, 
              int indegree, int sources[], int sourceweights[],
              int outdegree, int destinations[], int destweights[],
              MPI_Info info, int reorder, MPI_Comm *comm_dist_graph);
int MPIX_Dist_graph_create(MPI_Comm comm_old, int n, int nodes[], int degrees[], 
                          int targets[], int weights[], MPI_Info info,
                          int reorder, MPI_Comm *newcomm);
int MPIX_Dist_graph_neighbors_count(MPI_Comm comm, int *inneighbors, int *outneighbors, int *weighted);
int MPIX_Dist_graph_neighbors(MPI_Comm comm, int maxindegree, int sources[], 
                       int sourceweights[], int maxoutdegree, 
                       int destinations[], int destweights[]);
}

#define MPIX_UNWEIGHTED NULL
