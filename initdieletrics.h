#include <cuda.h>
#include <math.h>
#include <stdlib.h>
#include "geo2D.h"

float *e_h;

int square(int i, int j, int x, int y, int a, int b);
int ring(int i, int j, int x, int y, int a, int b);
int circle(int i, int j, int x, int y, int a);
void SETTING_DIELETRICS()
  {
  e_h = (float *)malloc(Dsize);
  
  int i,j;
  for(i=0;i<Lx;i++)
    {
    for(j=0;j<Ly;j++)
        {
	//e_h[i+Lx*j] = epso;
//	e_h[i+j*Lx] = epso + 5.f * epso * square(i,j,10,60,90,10) + 15.f * epso * ring(i,j,90,60,30,20) + 9.f * epso * circle(i,j,10,30,20)   ;
        e_h[i+j*Lx] = epso + 9.f * epso * ring(i,j,60,60,20,40)   ;
	}
    }
  }
  
  