#include <cuda.h>
#include <math.h>
#include <stdlib.h>
// #include "kernels-global.h"
  float *alpha_h_x;
  float *alpha_h_y;
  float *alpha_h_z;
  float *beta_h_x;
  float *beta_h_y;
  float *beta_h_z;
  
// __global__ void InitConstUPML(float *xp, float *yp, float*zp, float *xm, float *ym, float *zm,  int pml_lenghtx, int pml_lenghty, UPML_CONS *out);

void SETTING_UPML()
{
  /////////////////// PML properties ///////////////////////////
  int m = 3;
  //   rate_y = rate_y * rate_y * rate_y;
  float R = 1e-8;
  
  float sigma0_x;
  float sigma0_y;
  float kappa0_x;
  float kappa0_y;
  
  if ( pml_lx != 0)
    {
    sigma0_x = -( m + 1 ) * log (R) / ( 2.f * eta * dx * pml_lx ); 
    }
  else
    {
    sigma0_x=0.f;
    }
    
  if ( pml_ly != 0)
    {
    sigma0_y = -( m + 1 ) * log (R) / ( 2.f * eta * dy * pml_ly ); 
    }
  else
    {
    sigma0_y=0.f;
    } 

// ////////// Aloccate PML constants  //////// 
  float *kappa_x;
  float *kappa_y;
  float *kappa_z;
  float *sigma_x;
  float *sigma_y;
  float *sigma_z;
//   
//   float *alpha_x;
//   float *alpha_y;
//   float *alpha_z;
//   float *beta_x;
//   float *beta_y;
//   float *beta_z;
//   UPML_CONS *upml;
//   
//   float *alpha_h_x;
//   float *alpha_h_y;
//   float *alpha_h_z;
//   float *beta_h_x;
//   float *beta_h_y;
//   float *beta_h_z;
//   
//   cudaMalloc ((void **) &alpha_x, Dsize);
//   cudaMalloc ((void **) &alpha_y, Dsize);
//   cudaMalloc ((void **) &alpha_z, Dsize);
//   cudaMalloc ((void **) &beta_x, Dsize);
//   cudaMalloc ((void **) &beta_y, Dsize);
//   cudaMalloc ((void **) &beta_z, Dsize);
//   cudaMalloc ((void **) &upml, sizeof(UPML_CONS) );
//   
  alpha_h_x = (float *)malloc(Dsize);
  alpha_h_y = (float *)malloc(Dsize);
  alpha_h_z = (float *)malloc(Dsize);
  beta_h_x = (float *)malloc(Dsize);
  beta_h_y = (float *)malloc(Dsize);
  beta_h_z = (float *)malloc(Dsize);
  
  kappa_x = (float *)malloc(Dsize);
  kappa_y = (float *)malloc(Dsize);
  kappa_z = (float *)malloc(Dsize);
  sigma_x = (float *)malloc(Dsize);
  sigma_y = (float *)malloc(Dsize);
  sigma_z = (float *)malloc(Dsize);
  
////////////////////////////////////////////////////////////////////

float rate;
int i,j;

for(j=0;j<Ly;j++)
  {
  for(i=0;i<Lx;i++)
    {
    if (pml_lx > 0)
      {
      if ( i < pml_lx )  
	{
	rate = pow ( ( (float) (pml_lx - i - 1) ) / ( (float) pml_lx ), (float) m);  
	sigma_x[i+j*Lx] = rate * sigma0_x;
        kappa_x[i+j*Lx] = rate * kappa0_x + 1.f;
	}
	
      else if ( i > lx + pml_lx-1 )  
	{
	rate = pow ( ( (float) (i - (pml_lx + lx)) ) / ( (float) pml_lx ),(float) m);  
        sigma_x[i+j*Lx] = rate * sigma0_x;
        kappa_x[i+j*Lx] = rate * kappa0_x + 1.f;
	}
      else
	{
	sigma_x[i+j*Lx]=0.f;
	kappa_x[i+j*Lx]=1.f;
	}
      }
     else
	{
	sigma_x[i+j*Lx]=0.f;
	kappa_x[i+j*Lx]=1.f;
	}  
      
    if (pml_ly > 0)
      {
      if ( j < pml_ly )  
	{
	rate = pow ( ( (float) (pml_ly - j - 1) ) / ( (float) pml_ly ), (float) m);  
        sigma_y[i+j*Lx] = rate * sigma0_y;
        kappa_y[i+j*Lx] = rate * kappa0_y + 1.f;
	}
	
      else if ( j > ly + pml_ly-1 )  
	{
	rate = pow ( ( (float) (j - (pml_ly + ly)) ) / ( (float) pml_ly ),(float) m); 
        sigma_y[i+j*Lx] = rate * sigma0_y;
        kappa_y[i+j*Lx] =  rate * kappa0_y + 1.f;
	}
      else
	{
	sigma_y[i+j*Lx]=0.f;
	kappa_y[i+j*Lx]=1.f;
	}
      }   
      else
	{
	sigma_y[i+j*Lx]=0.f;
	kappa_y[i+j*Lx]=1.f;
	}
	
     sigma_z[i+j*Lx]=0.f;
     kappa_z[i+j*Lx]=1.f;
      
     alpha_h_x[i+j*Lx] = kappa_x[i+j*Lx]/dt + sigma_x[i+j*Lx] / (2.f* epso);
     alpha_h_y[i+j*Lx] = kappa_y[i+j*Lx]/dt + sigma_y[i+j*Lx] / (2.f* epso);
     alpha_h_z[i+j*Lx] = kappa_z[i+j*Lx]/dt + sigma_z[i+j*Lx] / (2.f* epso);
     beta_h_x[i+j*Lx] = kappa_x[i+j*Lx]/dt - sigma_x[i+j*Lx] / (2.f* epso);
     beta_h_y[i+j*Lx] = kappa_y[i+j*Lx]/dt - sigma_y[i+j*Lx] / (2.f* epso);
     beta_h_z[i+j*Lx] = kappa_z[i+j*Lx]/dt - sigma_z[i+j*Lx] / (2.f* epso);  
      
    }
  }  
 
//   cudaMemcpy (alpha_x, alpha_h_x, Dsize, cudaMemcpyHostToDevice);
//   cudaMemcpy (alpha_y, alpha_h_y, Dsize, cudaMemcpyHostToDevice);
//   cudaMemcpy (alpha_z, alpha_h_z, Dsize, cudaMemcpyHostToDevice);
//   cudaMemcpy (beta_x, beta_h_x, Dsize, cudaMemcpyHostToDevice);
//   cudaMemcpy (beta_y, beta_h_y, Dsize, cudaMemcpyHostToDevice);
//   cudaMemcpy (beta_z, beta_h_z, Dsize, cudaMemcpyHostToDevice);

//   InitConstUPML <<< 1, 1 >>> (alpha_x, alpha_y, alpha_z, beta_x, beta_y, beta_z, pml_lx, pml_ly, upml);
  
//   free(alpha_h_x);
//   free(alpha_h_y);
//   free(alpha_h_z);
//   free(beta_h_x);
//   free(beta_h_y);
//   free(beta_h_z);
   free(kappa_x);
   free(kappa_y);
   free(kappa_z);
   free(sigma_x);
   free(sigma_y);
   free(sigma_z);
}