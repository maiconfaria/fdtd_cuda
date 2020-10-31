#include <cuda.h>
#include <math.h>
#include <stdlib.h>
   
int square(int i, int j, int x, int y, int a, int b)
  {
  int n; 
  if(i>x && i<x+a && j>y && j<y+b)
    {
    n=1;
    }
  else
    {
    n=0;
    }
  return n; 
  }
   
int ring(int i, int j, int x, int y, int a, int b)
  {
  int n;
  int tmp;
  tmp=(i-x)*(i-x)+(j-y)*(j-y);
    
  if(tmp>a*a && tmp<b*b)
    {
    n=1;
    }
  else
    {
    n=0;
    }
  return n; 
  }
  
int circle(int i, int j, int x, int y, int a)
  {
  int n;
  int tmp;
  tmp=(i-x)*(i-x)+(j-y)*(j-y);
    
  if(tmp<a*a)
    {
    n=1;
    }
  else
    {
    n=0;
    }
  return n; 
  }  