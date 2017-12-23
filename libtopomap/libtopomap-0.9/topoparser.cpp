#include "libtopomap.hpp"

static int read_int(unsigned char *t, int *pos) {
  int num=0;
  const int radix=10;
   
  while(*t >= '0' && *t <= '9') {
    num = num * radix + (*t - (unsigned char)'0');
    t++;
    (*pos)++;
  }
  
  return num;
}

/* the file format is extremely simplistic:
 * line 1:                      num: <num vertices>
 * line 2 to <num vertices>+1:  <vertex num> <vertex name>
 * line <num vertices>+1 to 2*<num vertices>+1 : <vertex num> <adj vertex 1> ... <adj vertex n>
 *
 * vertices can occur in any order! Example:
--- snip 
num: 3
1 blub
0 bla
2 blib
1 2 0
2 0 1
0 1 2
--- snip
 */

int TPM_Read_topo(const char *topofile, TPM_Nodenames *names, TPM_Graph *g, TPM_Graph_ewgts *w) {
  // open topology file
  FILE *fd = fopen(topofile, "r");
  if (!fd) return 1; // fopen failed

  struct stat statbuf;
  stat(topofile, &statbuf);
  
  // read complete file in memory ...
  unsigned char *buf=(unsigned char*)malloc(statbuf.st_size);;
  int ret = fread(buf, 1, statbuf.st_size, fd);
  fclose(fd);
  int pos=0;
  // scan to first number
  while(! (buf[pos] >= '0' && buf[pos] <= '9') ) pos++;
  int n = read_int(&buf[pos], &pos);

  int weighted=0;
  while(buf[pos] != '\n') {
    if(buf[pos]=='w') weighted=1;
    pos++; 
  }
  pos++;

  DBG(printf("n=%i w=%i\n", n, weighted));

  int i;
  names->resize(n);
#define NSIZE 1024
  char name[1024];
  for(i=0; i<n; i++) {
    int v = read_int(&buf[pos], &pos);

    while(buf[pos] == ' ') pos++; // fast-forward over whites

    int npos=0;
    while(buf[pos] != '\n') {
      name[npos++] = buf[pos++];
      if(npos>NSIZE) { printf("name overflow\n"); return 0; }
    }
    name[npos] = '\0';
  
    while(buf[pos] != '\n') pos++; pos++; // next line

    DBG(printf("%i >%s<\n", v, name));
    (*names)[v] = name;
  }

  g->resize(n);
  w->resize(n);
  for(i=0; i<n; i++) {
    int u = read_int(&buf[pos], &pos);
    DBG(printf("new vertex: %i, adj. list:", u));

    while(buf[pos] != '\n') {
      while(buf[pos] == ' ') pos++; // fast-forward over whites
      int v = read_int(&buf[pos], &pos);
      (*g)[u].push_back(v);
    
      int wgt = 1;
      // weighted graphs have a weight in "()" without a space!!
      if(weighted) {
        if(buf[pos] != '(') printf("format error!\n");
        pos++; 
        wgt = read_int(&buf[pos], &pos);
        if(buf[pos] != ')') printf("format error!\n");
        pos++; 
      }
      (*w)[u].push_back(wgt);

      DBG(printf(" %i(%i)", v, wgt));
    }
    DBG(printf("\n"));
    pos++; // skip over \n
  }
  free(buf);
  return 0;
}


int TPM_Fake_hostname(const char *fname, int rank, char *name, int namesize) {
  struct stat sbuf;
  stat(fname, &sbuf);

  FILE *f = fopen(fname, "r");
  
  unsigned char *buf= (unsigned char*)malloc(sbuf.st_size);
  int ret = fread(buf, 1, sbuf.st_size, f);
  int pos=0;

#define rint register int 
  int v=-1;
  while(pos < sbuf.st_size && v != rank) {
    v = read_int(&buf[pos], &pos);

    while(buf[pos] == ' ') pos++; // fast-forward over whites

    int npos=0;
    while(buf[pos] != '\n') {
      name[npos++] = buf[pos++];
      if(npos>namesize) { printf("name overflow\n"); return 0; }
    }
    name[npos] = '\0';
  
    while(buf[pos] != '\n') pos++; pos++; // next line
  }

  free(buf);
  fclose(f);
  return 0;
}
