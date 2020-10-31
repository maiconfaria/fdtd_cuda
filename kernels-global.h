 #include "initparam.h"




typedef struct 
  {
        float *xp;
        float *yp;
        float *zp;
	float *xm;
        float *ym;
        float *zm;
	int pml_lenghtx;
	int pml_lenghty;
  } UPML_CONS;

typedef struct 
  {
        float *x;
        float *y;
  } Surface;
  
typedef struct 
  {
        float gdt;
        float gdx;
        float gdy;
	int glx;
        int gly;
        int gLx;
	int gLy;
  } GEO_CONS;  
  


__global__ void InitConstGEO(float gdt, float gdx, float gdy, int glx, int gly, int gLx, int gLy, GEO_CONS *out)
  {
  out->gdt = gdt;
  out->gdx = gdx;
  out->gdy = gdy;
  out->glx = glx;
  out->gly = gly;
  out->gLx = gLx;
  out->gLy = gLy;
  }
 
__global__ void InitConstUPML(float *xp, float *yp, float*zp, float *xm, float *ym, float *zm, int pml_lenghtx, int pml_lenghty, UPML_CONS *out)
  {
  out->xp = xp;
  out->yp = yp;
  out->zp = zp;
  out->xm = xm;
  out->ym = ym;
  out->zm = zm;
  out->pml_lenghtx = pml_lenghtx;
  out->pml_lenghty = pml_lenghty;
  }
 
 __global__ void InitConstConductors(float *x, float *y, Surface *out)
  {
  out->x = x;
  out->y = y;
  }

 
 
 __global__ void WaveStepE (float *Ex, float *Dx, float *Ey, float *Dy, float *Hz, float *Bz, float *ez, UPML_CONS *upml, float *out, Surface *csur, GEO_CONS *geo)
  {
  int i = blockIdx.x * (blockDim.x) + threadIdx.x;
  int j = blockIdx.y * (blockDim.y) + threadIdx.y;
  int k = blockIdx.z * (blockDim.z) + threadIdx.z;
  
  float dt = geo->gdt;
  float dx = geo->gdx;
  float dy = geo->gdy;
  int lx = geo->glx;
  int ly = geo->gly;
  int Lx = geo->gLx;
  int Ly = geo->gLy;
  
  
  float epsilon = ez[i+j*Lx];

  float DxStore = Dx[i+j*Lx];
  float DyStore = Dy[i+j*Lx];
  
  float alpha_x = upml->xp[i+j*Lx];
  float alpha_y = upml->yp[i+j*Lx];
  float alpha_z = upml->zp[i+j*Lx];
  float beta_x = upml->xm[i+j*Lx];
  float beta_y = upml->ym[i+j*Lx];
  float beta_z = upml->zm[i+j*Lx];
  
  int pml_lx = upml->pml_lenghtx;
  int pml_ly = upml->pml_lenghty;
    
  
  float csur_x = csur->x[i+j*Lx];
  float csur_y = csur->y[i+j*Lx];
  
   if( j < (ly + 2 * pml_ly - 1)  && i < (lx + 2 * pml_lx - 1) ) 
    {
        
    Dx[i+j*Lx] = ( beta_y / alpha_y ) * Dx[i+j*Lx] - ( 1.f / ( dy * alpha_y ) ) * (Hz[i+(j+1)*Lx] - Hz[i+j*Lx]);
    
    Ex[i+j*Lx] = csur_x * ( ( beta_z / alpha_z ) * Ex[i+j*Lx] + ( alpha_x / ( epsilon * alpha_z ) ) * Dx[i+j*Lx] - ( beta_x / ( epsilon * alpha_z ) ) * DxStore );
  
    
    
    Dy[i+j*Lx] = ( beta_z / alpha_z ) * Dy[i+j*Lx] + ( 1.f / ( dx * alpha_z ) ) * (Hz[i+1+j*Lx] - Hz[i+j*Lx]);
    
    Ey[i+j*Lx] = csur_y *  ( ( beta_x / alpha_x ) * Ey[i+j*Lx] + ( alpha_y / ( epsilon * alpha_x ) ) *  Dy[i+j*Lx] - ( beta_y / ( epsilon * alpha_x ) ) * DyStore );
       
    }
  
   
//   out[i+j*Lx] =    ( beta_x / alpha_x );
  }
  
__global__ void WaveStepH (float *Ex, float *Dx, float *Ey, float *Dy, float *Hz, float *Bz, float field, float *ez, UPML_CONS *upml, float *out, float mu, GEO_CONS *geo)	
  {
      
  int i = blockIdx.x * (blockDim.x) + threadIdx.x;
  int j = blockIdx.y * (blockDim.y) + threadIdx.y;

  float dt = geo->gdt;
  float dx = geo->gdx;
  float dy = geo->gdy;
  int lx = geo->glx;
  int ly = geo->gly;
  int Lx = geo->gLx;
  int Ly = geo->gLy;

  
//   float epsilon = ez[i+j*Lx];
  float alpha_x = upml->xp[i+j*Lx];
  float alpha_y = upml->yp[i+j*Lx];
  float alpha_z = upml->zp[i+j*Lx];
  float beta_x = upml->xm[i+j*Lx];
  float beta_y = upml->ym[i+j*Lx];
  float beta_z = upml->zm[i+j*Lx];
  
  int pml_lx = upml->pml_lenghtx;
  int pml_ly = upml->pml_lenghty;
  

  
  float BzStore =  Bz[i+j*Lx];
  
   if(  j  > 0 && i > 0 ) 
   {
       
   Bz[i+j*Lx] =  ( beta_x / alpha_x ) * Bz[i+j*Lx] + ( ( 1.f / ( dx * alpha_x ) ) * (Ey[i+j*Lx] - Ey[i-1+j*Lx])  - ( 1.f / ( dy * alpha_x ) ) * (Ex[i+j*Lx] - Ex[i+(j-1)*Lx]) );

   Hz[i+j*Lx] =  ( beta_y / alpha_y ) * Hz[i+j*Lx] + ( alpha_z / ( mu * alpha_y ) ) * Bz[i+j*Lx] - ( beta_z / ( mu * alpha_y ) ) * BzStore;
     
   Hz[(pml_lx+lx/2)+(pml_ly+ly/2)*Lx]=field;
   }
   
     

   out[i+j*Lx] =  Hz[i+j*Lx]  ; 

  }


  void checkCUDAError (const char *msg)
  {
    cudaError_t err = cudaGetLastError ();
    if (cudaSuccess != err)
      {
	fprintf (stderr, "Cuda error: %s: %s.\n", msg,
		 cudaGetErrorString (err));
	exit (EXIT_FAILURE);
      }
  }

  
    
    void dprint (float *campo, int x, int y, int Lx)
    {
    for(int j = 0; j < y; j++)
      {
      for(int i = 0; i < x; i++)
	  {
// 	  if (campo[i+j*Lx]!=0.f)
	    {
	    printf("%d %d %f\n",i,j,campo[i+j*Lx]);
	    }
	  }
      }
    }
    
    void mprint (float *campo, int x, int y, int Lx )
    {
    for(int j = 0;j < y; j++)
      {
      for(int i = 0; i < x; i++)
	  {
	  printf("%g ", campo[i+j*Lx]);
	  }
      printf("\n");    
      }
    printf("\n");  
    }
    
    void midprint (float *campo, int Lx, int Ly)
    {
    printf("%f ", campo[Lx*(Ly/2)+Lx/2]);
    printf("\n");  
    }


