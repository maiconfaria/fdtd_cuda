#include <cuda.h>
#include <math.h>
#include <stdlib.h>

   float *csur_h_x; // 0 represents conductor and 1 not conductor.
   float *csur_h_y;

void SETTING_CONDUCTORS()
  {
  csur_h_x = (float *)malloc(Dsize);
  csur_h_y = (float *)malloc(Dsize);
  
  int i,j;
  
   for(i=0;i<Lx;i++)
    {
    for(j=0;j<Ly;j++)
      {
	csur_h_x[i+j*Lx] = 1.f;
	csur_h_y[i+j*Lx] = 1.f;  
      }
    }
 for(i=pml_lx;i<Lx-pml_lx;i++)
    {
    for(j=pml_ly;j<Ly-pml_lx;j++)
        {
// 	csur_h_x[i+j*Lx] = 1.f * (j!=10); //diagonal
// 	csur_h_y[i+j*Lx] = 1.f * (i!=j+12);
 //	csur_h_x[i+j*Lx] = 1.f * (j!=50 && j!=70);    
// 	csur_h_y[i+j*Lx] = 1.f * (i!=j-12);
	}
    }
  } 
    
   
