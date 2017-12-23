#include "libtopomap.hpp"

#ifdef BGP
#include <dcmf.h>

int TPM_Get_bgp_topo(int r, TPM_Graph *g_ptr, int *me) {
  TPM_Graph &g=*g_ptr;
  
  int nx, ny, nz; // maximum size per dimension
  int tx, ty, tz; // torus in x, y, z? 1 or 0!
  int x,y,z; // actual coordinates of rank

  DCMF_Hardware_t bgp_hw;
  DCMF_Hardware(&bgp_hw);
  nx=bgp_hw.xSize;
  ny=bgp_hw.ySize;
  nz=bgp_hw.zSize;
  tx=bgp_hw.xTorus;
  ty=bgp_hw.yTorus;
  tz=bgp_hw.zTorus;

  if(!r) printf("[%i] allocation size: %ix%ix%x %i ppn; torus: %i,%i,%i,%i\n", r, bgp_hw.xSize, bgp_hw.ySize, bgp_hw.zSize, bgp_hw.tSize, bgp_hw.xTorus, bgp_hw.yTorus, bgp_hw.zTorus, bgp_hw.tTorus);
        
  DCMF_NetworkCoord_t addr;
  DCMF_Messager_rank2network(r, DCMF_DEFAULT_NETWORK, &addr);
//if(!r) printf("[%i] coordinates: %i,%i,%i,%i\n", r, addr.torus.x, addr.torus.y, addr.torus.z, addr.torus.t);
  x=addr.torus.x;
  y=addr.torus.y;
  z=addr.torus.z;

  g.resize(nx*ny*nz);

  for(register int ix=0; ix<nx; ix++) {
    for(register int iy=0; iy<ny; iy++) {
      for(register int iz=0; iz<nz; iz++) {
        register int pos = iz+nz*(iy + ny*ix);
        if(x==ix && y==iy && z==iz) *me=pos;
        // build graph
        g[pos].push_back(iz+nz*(iy + ny*((ix+nx+1)%nx)));
        g[pos].push_back(iz+nz*(iy + ny*((ix+nx-1)%nx)));
        g[pos].push_back(iz+nz*(((iy+ny+1)%ny) + ny*ix));
        g[pos].push_back(iz+nz*(((iy+ny-1)%ny) + ny*ix));
        g[pos].push_back(((iz+nz+1)%nz)+nz*(iy + ny*ix));
        g[pos].push_back(((iz+nz-1)%nz)+nz*(iy + ny*ix));
      }
    }
  }

  return g.size();
}

#endif
