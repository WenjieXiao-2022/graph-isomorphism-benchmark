#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <set>

/*
 * Translate unlabeled, directed graphs in the binary format described at
 * http://amalfi.dis.unina.it/graph/db/doc/graphdb-3.html
 * to the DIMACS variant accepted by the bliss tool available at
 * http://www.tcs.hut.fi/Software/bliss
 */

/*
 * Copyright (c) Tommi Junttila
 * Released under the GNU General Public License version 2.
 */

static unsigned int read_word(FILE *fp)
{
  unsigned char b1 = getc(fp); /* Least-significant Byte */
  unsigned char b2 = getc(fp); /* Most-significant Byte */
  return(b1 | (b2 << 8));
}

int main()
{
  const bool verbose = false;
  FILE * const verbstr = stdout;
  
  FILE *in = stdin;
  FILE *out = stdout;
  
  /* Read the number of nodes */
  const unsigned int nof_vertices = read_word(in);
  
  if(nof_vertices <= 0)
    {
      fprintf(stderr, "error: no vertices, aborting\n");
      exit(1);
    }
  
  std::vector<std::set<unsigned int, std::less<unsigned int> > > edges = std::vector<std::set<unsigned int, std::less<unsigned int> > >(nof_vertices);
  
  /* For each vertex v... */
  for(unsigned int v = 0; v < nof_vertices; v++)
    { 
      /* Read the number of edges coming out of vertex v */
      const unsigned int outdegree = read_word(in);
      
      /* For each edge out of vertex v... */
      for(unsigned int e = 0; e < outdegree; e++)
	{ 
	  /* Read the destination node of the edge */
	  const unsigned int target = read_word(in);	  
	  if(target >= nof_vertices)
	    {
	      fprintf(stderr, "Error: the vertex %u is not in "
		      "the range [0,...,%u], aborting\n",
		      target, nof_vertices-1);
	      exit(1);
	    }
	  edges[v].insert(target);
	}
    }
  
  unsigned int nof_edges = 0;
  for(unsigned int v = 0; v < nof_vertices; v++)
    nof_edges += edges[v].size();
  
  if(verbose)
    {
      fprintf(verbstr, "Instance has %d vertices and %d edges\n",
	      nof_vertices, nof_edges);
      fflush(verbstr);
    }
  
  fprintf(out, "p edge %u %u\n", nof_vertices, nof_edges);
  
  for(unsigned int v = 0; v < nof_vertices; v++)
    for(std::set<unsigned int, std::less<unsigned int> >::const_iterator ei = edges[v].begin(); ei != edges[v].end(); ei++)
      fprintf(out, "e %u %u\n", v+1, *ei+1);
  
  return 0;
}
