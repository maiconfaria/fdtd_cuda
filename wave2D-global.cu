#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <unistd.h>
#include "kernels-global.h"
#include "initparam.h"
#include "upml.h"
#include "initconductors.h"
#include "initdieletrics.h"
// #include "geo2D.h"
// #define BLOCKSIZEX 16
// #define BLOCKSIZEY 16
     

 void checkCUDAError (const char *msg);

 void dprint (float *campo, int x, int y, int Lx);
 void mprint (float *campo, int x, int y, int Lx);
 void midprint (float *campo, int Lx, int Ly);

// void SETTING_UPML();
 __global__ void InitConstUPML(float *xp, float *yp, float*zp, float *xm, float *ym, float *zm,  int pml_lenghtx, int pml_lenghty, UPML_CONS *out);
 __global__ void InitConstConductors(float *x, float *y, Surface *out);
 __global__ void InitConstGEO(float gdt, float gdx, float gdy, int glx, int gly, int gLx, int gLy, GEO_CONS *out);
 __global__ void WaveStepE (float *Ex, float *Dx, float *Ey, float *Dy, float *Hz, float *Bz, float *e , UPML_CONS *upml, float *out, Surface *csur, GEO_CONS *geo);
 __global__ void WaveStepH (float *Ex, float *Dx, float *Ey, float *Dy, float *Hz, float *Bz, float field, float *e , UPML_CONS *upml, float *out, float mu, GEO_CONS *geo);	

 
 
int main (int argc, char **argv)
  {
      
  int i,j;  
  /////////// Set computational geometric Sizes ///////////////////
  GEO_CONS *geo;                                                 //
  cudaMalloc ((void **) &geo, sizeof(GEO_CONS) );                //
  InitConstGEO <<< 1, 1 >>> (dt, dx, dy, lx, ly, Lx, Ly, geo);   //
  /////////////////////////////////////////////////////////////////  
  
  ////////// Eletrical permittivity and magnetic permeability//////// 
  float *e ;                                                       //
  cudaMalloc ((void **) &e , Dsize);                               //
  SETTING_DIELETRICS();                                            //
  cudaMemcpy (e , e_h, Dsize, cudaMemcpyHostToDevice);             //
  ///////////////////////////////////////////////////////////////////
  
  
  ////////// Aloccate PML constants  ///////////////////////////////////////////////////////////////////////// 
  float *alpha_x;                                                                                           //
  float *alpha_y;                                                                                           //
  float *alpha_z;                                                                                           //
  float *beta_x;                                                                                            //
  float *beta_y;                                                                                            //
  float *beta_z;                                                                                            //
  UPML_CONS *upml;                                                                                          //
                                                                                                            //
  cudaMalloc ((void **) &alpha_x, Dsize);                                                                   //
  cudaMalloc ((void **) &alpha_y, Dsize);                                                                   //
  cudaMalloc ((void **) &alpha_z, Dsize);                                                                   //
  cudaMalloc ((void **) &beta_x, Dsize);                                                                    //
  cudaMalloc ((void **) &beta_y, Dsize);                                                                    //
  cudaMalloc ((void **) &beta_z, Dsize);                                                                    //
  cudaMalloc ((void **) &upml, sizeof(UPML_CONS) );                                                         //
                                                                                                            //
  SETTING_UPML();                                                                                           //
                                                                                                            //
  cudaMemcpy (alpha_x, alpha_h_x, Dsize, cudaMemcpyHostToDevice);                                           //
  cudaMemcpy (alpha_y, alpha_h_y, Dsize, cudaMemcpyHostToDevice);                                           //
  cudaMemcpy (alpha_z, alpha_h_z, Dsize, cudaMemcpyHostToDevice);                                           //
  cudaMemcpy (beta_x, beta_h_x, Dsize, cudaMemcpyHostToDevice);                                             //
  cudaMemcpy (beta_y, beta_h_y, Dsize, cudaMemcpyHostToDevice);                                             //
  cudaMemcpy (beta_z, beta_h_z, Dsize, cudaMemcpyHostToDevice);                                             //
                                                                                                            //
  InitConstUPML <<< 1, 1 >>> (alpha_x, alpha_y, alpha_z, beta_x, beta_y, beta_z, pml_lx, pml_ly, upml);     //
                                                                                                            //
  free(alpha_h_x);                                                                                          //
  free(alpha_h_y);                                                                                          //
  free(alpha_h_z);                                                                                          //
  free(beta_h_x);                                                                                           //
  free(beta_h_y);                                                                                           //
  free(beta_h_z);                                                                                           //
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  
  
  /////////Aloccate eletrical magnetic fields arrays///////////////
  float *Ex_h;                                                   //                                                                       
  float *Ey_h;                                                   //
  float *Hz_h;                                                   //
  float *Dx_h;                                                   //
  float *Dy_h;                                                   //
  float *Bz_h;                                                   //
                                                                 //
  Ex_h = (float *)malloc(Dsize);                                 //
  Ey_h = (float *)malloc(Dsize);                                 //
  Hz_h = (float *)malloc(Dsize);                                 //
  Dx_h = (float *)malloc(Dsize);                                 //
  Dy_h = (float *)malloc(Dsize);                                 //
  Bz_h = (float *)malloc(Dsize);                                 //
                                                                 //
    /////////// Null Field initial condition //////              //
                                                                 //
  for(i=0;i<Lx;i++)                                              //
    {                                                            //
    for(j=0;j<Ly;j++)                                            //
        {                                                        //
        Ex_h[i+Lx*j]=0.f;                                        //
        Ey_h[i+Lx*j]=0.f;                                        //
        Hz_h[i+Lx*j]=0.f;                                        //
	Dx_h[i+Lx*j]=0.f;                                        //
        Dy_h[i+Lx*j]=0.f;                                        //
        Bz_h[i+Lx*j]=0.f;                                        //
        }                                                        //
    }                                                            //
                                                                 //
  float *Ex;                                                     //
  float *Ey;                                                     //
  float *Hz;                                                     //
  float *Dx;                                                     //
  float *Dy;                                                     //
  float *Bz;                                                     //
                                                                 //
  cudaMalloc ((void **) &Ex, Dsize);                             //
  cudaMalloc ((void **) &Ey, Dsize);                             //
  cudaMalloc ((void **) &Hz, Dsize);                             //
  cudaMalloc ((void **) &Dx, Dsize);                             //
  cudaMalloc ((void **) &Dy, Dsize);                             //
  cudaMalloc ((void **) &Bz, Dsize);                             //
                                                                 //
    /////////////// Coping data to Device /////////////////////  //
  cudaMemcpy (Ex, Ex_h, Dsize, cudaMemcpyHostToDevice);          //
  cudaMemcpy (Ey, Ey_h, Dsize, cudaMemcpyHostToDevice);          //
  cudaMemcpy (Hz, Hz_h, Dsize, cudaMemcpyHostToDevice);          //
  cudaMemcpy (Dx, Dx_h, Dsize, cudaMemcpyHostToDevice);          //
  cudaMemcpy (Dy, Dy_h, Dsize, cudaMemcpyHostToDevice);          //
  cudaMemcpy (Bz, Bz_h, Dsize, cudaMemcpyHostToDevice);          //
  /////////////////////////////////////////////////////////////////

    
  
  
 
  
//   cudaMemcpy (e , e_h, Dsize, cudaMemcpyHostToDevice);
  ///////////////////////////////////////////////////////////
  /////////////////////////// Allocating conductors ///////////////////
                                                                     //
  float *csur_x;                                                     //
  float *csur_y;                                                     //
  Surface *csur;                                                     //
                                                                     //
  cudaMalloc ((void **) &csur_x, Dsize);                             //
  cudaMalloc ((void **) &csur_y, Dsize);                             //
  cudaMalloc ((void **) &csur, sizeof(Surface) );                    //
                                                                     //
  SETTING_CONDUCTORS();                                              //
                                                                     //
  cudaMemcpy (csur_x, csur_h_x, Dsize, cudaMemcpyHostToDevice);      //
  cudaMemcpy (csur_y, csur_h_y, Dsize, cudaMemcpyHostToDevice);      //
                                                                     //
  InitConstConductors <<< 1, 1 >>> (csur_x,csur_y, csur);            //
  free(csur_h_x);                                                    //
  free(csur_h_y);                                                    //
 //////////////////////////////////////////////////////////////////////
 
  
  ////////// Debbuger out array ////////// 
  float *out_h;                         //
  out_h = (float *)malloc(Dsize);       //
  float *out;                           //
  cudaMalloc ((void **) &out, Dsize);   //
  ///////////////////////////////////////
  
  
  
  
  ////////////////////Time iteration ///////////////////////////////////////////////////////////////////////
  int T=4000;                                                                                             //
                                                                                                          //
  int b = 25.0;                                                                                           //
  float dum,voltage,field;                                                                                //
  for (int t = 1; t < T; t = t + 1)	//iterando no tempo                                               //
    {                                                                                                     //
                                                                                                          //
                                                                                                          //
    checkCUDAError ("getting data from device");                                                          //
                                                                                                          //
      dum = (4.0/b/dt)*(t*dt-b*dt);                                                                       //
      voltage = 2.0*dum*exp(-(pow(dum,2.f)));                                                             //
//       if ( t < 5 )                                                                                     //
	{                                                                                                 //
	//field = voltage / (sqrt(co)*dx);                                                                  //
	field = voltage / dx;
	}                                                                                                 //
//       else                                                                                             //
// 	{                                                                                                 //
// 	field = 0.f;                                                                                      //
// 	}                                                                                                 //
                                                                                                          //
    WaveStepH <<< dimGrid, dimBlock >>> (Ex, Dx, Ey, Dy, Hz, Bz, field, e , upml, out, muo, geo);         //
    WaveStepE <<< dimGrid, dimBlock >>> (Ex, Dx, Ey, Dy, Hz, Bz, e , upml, out, csur, geo);               //
                                                                                                          //
    checkCUDAError ("kernel invocation");                                                                 //
    cudaMemcpy (Hz_h, Hz, Dsize, cudaMemcpyDeviceToHost);                                            //
                                            //
                                                                                                          //
    checkCUDAError ("getting data from device");                                                          //
//    dprint (Hz_h, lx + 2 * pml_lx, ly + 2 * pml_ly, Lx);                                        //
    }                                                                                                     //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////  
}

