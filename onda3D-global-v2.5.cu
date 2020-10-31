#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <unistd.h>

#define BLOCKSIZEX 8
#define BLOCKSIZEY 8
#define BLOCKSIZEZ 8

 void checkCUDAError (const char *msg);
 void dprint (float *campo, int x, int y, int Lx, int Ly, int Lz);
 void mprint (float *campo, int x, int y, int Lx, int Ly, int Lz);
 void midprint (float *campo, int Lx, int Ly, int Lz);
 void cria_arquivo(float *campo, int x, int y,int t, int Lx, int Ly, int Lz);

__global__ void WaveStepH (float *Hx, float *Hy, float *Hz, float *Ex, float *Ey, float *Ez, int Lx, int Ly, int Lz, float mx, float my, float *out)
      {
         int i = blockIdx.x * (blockDim.x) + threadIdx.x;
	 int j = blockIdx.y * (blockDim.y) + threadIdx.y;
	 int k = blockIdx.z * (blockDim.z) + threadIdx.z;
	 if( j< Ly && i<Lx && k<Lz)
             {
	       
                             
		Hx[4*i+j*Lx+k*Lx*Ly] = Hx[4*i+j*Lx+k*Lx*Ly] + mx * (Ey[4*i+j*Lx+(k+1)*Lx*Ly] - Ey[4*i+j*Lx+k*Lx*Ly] - Ez[4*i+(j+1)*Lx+k*Lx*Ly] + Ez[4*i+j*Lx+k*Lx*Ly] );
                Hx[(4*i+1)+j*Lx+k*Lx*Ly] = Hx[(4*i+1)+j*Lx+k*Lx*Ly] + mx * (Ey[(4*i+1)+j*Lx+(k+1)*Lx*Ly] - Ey[(4*i+1)+j*Lx+k*Lx*Ly] - Ez[(4*i+1)+(j+1)*Lx+k*Lx*Ly] + Ez[(4*i+1)+j*Lx+k*Lx*Ly] );
		Hx[(4*i+2)+j*Lx+k*Lx*Ly] = Hx[(4*i+2)+j*Lx+k*Lx*Ly] + mx * (Ey[(4*i+2)+j*Lx+(k+1)*Lx*Ly] - Ey[(4*i+2)+j*Lx+k*Lx*Ly] - Ez[(4*i+2)+(j+1)*Lx+k*Lx*Ly] + Ez[(4*i+2)+j*Lx+k*Lx*Ly] );
		Hx[(4*i+3)+j*Lx+k*Lx*Ly] = Hx[(4*i+3)+j*Lx+k*Lx*Ly] + mx * (Ey[(4*i+3)+j*Lx+(k+1)*Lx*Ly] - Ey[(4*i+3)+j*Lx+k*Lx*Ly] - Ez[(4*i+3)+(j+1)*Lx+k*Lx*Ly] + Ez[(4*i+3)+j*Lx+k*Lx*Ly] );
		
		Hy[4*i+j*Lx+k*Lx*Ly] = Hy[4*i+j*Lx+k*Lx*Ly] + mx * (Ez[4*i+1+j*Lx+k*Lx*Ly] - Ez[4*i+j*Lx+k*Lx*Ly] - Ex[4*i+j*Lx+(k+1)*Lx*Ly] + Ex[4*i+j*Lx+k*Lx*Ly] );
                Hy[(4*i+1)+j*Lx+k*Lx*Ly] = Hy[(4*i+1)+j*Lx+k*Lx*Ly] + mx * (Ez[(4*i+2)+j*Lx+k*Lx*Ly] - Ez[(4*i+1)+j*Lx+k*Lx*Ly] - Ex[(4*i+1)+j*Lx+(k+1)*Lx*Ly] + Ex[(4*i+1)+j*Lx+k*Lx*Ly] );
		Hy[(4*i+2)+j*Lx+k*Lx*Ly] = Hy[(4*i+2)+j*Lx+k*Lx*Ly] + mx * (Ez[(4*i+3)+j*Lx+k*Lx*Ly] - Ez[(4*i+2)+j*Lx+k*Lx*Ly] - Ex[(4*i+2)+j*Lx+(k+1)*Lx*Ly] + Ex[(4*i+2)+j*Lx+k*Lx*Ly] );
		Hy[(4*i+3)+j*Lx+k*Lx*Ly] = Hy[(4*i+3)+j*Lx+k*Lx*Ly] + mx * (Ez[(4*i+4)+j*Lx+k*Lx*Ly] - Ez[(4*i+3)+j*Lx+k*Lx*Ly] - Ex[(4*i+3)+j*Lx+(k+1)*Lx*Ly] + Ex[(4*i+3)+j*Lx+k*Lx*Ly] );
		
		Hz[4*i+j*Lx+k*Lx*Ly] = Hz[4*i+j*Lx+k*Lx*Ly] + mx * (Ex[4*i+(j+1)*Lx+k*Lx*Ly] - Ex[4*i+j*Lx+k*Lx*Ly] - Ey[4*i+1+j*Lx+k*Lx*Ly] + Ey[4*i+j*Lx+k*Lx*Ly] );
                Hz[(4*i+1)+j*Lx+k*Lx*Ly] = Hz[(4*i+1)+j*Lx+k*Lx*Ly] + mx * (Ex[(4*i+1)+(j+1)*Lx+k*Lx*Ly] - Ex[(4*i+1)+j*Lx+k*Lx*Ly] - Ey[(4*i+2)+j*Lx+k*Lx*Ly] + Ey[(4*i+1)+j*Lx+k*Lx*Ly] );
		Hz[(4*i+2)+j*Lx+k*Lx*Ly] = Hz[(4*i+2)+j*Lx+k*Lx*Ly] + mx * (Ex[(4*i+2)+(j+1)*Lx+k*Lx*Ly] - Ex[(4*i+2)+j*Lx+k*Lx*Ly] - Ey[(4*i+3)+j*Lx+k*Lx*Ly] + Ey[(4*i+2)+j*Lx+k*Lx*Ly] );
		Hz[(4*i+3)+j*Lx+k*Lx*Ly] = Hz[(4*i+3)+j*Lx+k*Lx*Ly] + mx * (Ex[(4*i+3)+(j+1)*Lx+k*Lx*Ly] - Ex[(4*i+3)+j*Lx+k*Lx*Ly] - Ey[(4*i+4)+j*Lx+k*Lx*Ly] + Ey[(4*i+3)+j*Lx+k*Lx*Ly] );
		
		
             }
      }
      
__global__ void WaveStepE (float *Hx, float *Hy, float *Hz, float *Ex, float *Ey, float *Ez, int Lx, int Ly,int Lz, float field, float *ez, float *out)
      {
         int i = blockIdx.x * (blockDim.x) + threadIdx.x;
	 int j = blockIdx.y * (blockDim.y) + threadIdx.y;
	 int k = blockIdx.z * (blockDim.z) + threadIdx.z;
     //    if( j>1 && i>1 && k>1 && j<Ly-1 && i<Lx-1 && k<Lz-1)
	  if( j>0 && i>0 && k>0 ) 
                {   
                
		   
		    
		    
		    Ex[4*i+j*Lx+k*Lx*Ly] = Ex[4*i+j*Lx+k*Lx*Ly] + ez[4*i+j*Lx+k*Lx*Ly] * (Hz[4*i+j*Lx+k*Lx*Ly] - Hz[4*i+(j-1)*Lx+k*Lx*Ly] - Hy[4*i+j*Lx+k*Lx*Ly] + Hy[4*i+j*Lx+(k-1)*Lx*Ly] );
		    Ex[(4*i+1)+j*Lx+k*Lx*Ly] = Ex[(4*i+1)+j*Lx+k*Lx*Ly] + ez[(4*i+1)+j*Lx+k*Lx*Ly] * (Hz[(4*i+1)+j*Lx+k*Lx*Ly] - Hz[(4*i+1)+(j-1)*Lx+k*Lx*Ly] - Hy[(4*i+1)+j*Lx+k*Lx*Ly] + Hy[(4*i+1)+j*Lx+(k-1)*Lx*Ly] );
		    Ex[(4*i+2)+j*Lx+k*Lx*Ly] = Ex[(4*i+2)+j*Lx+k*Lx*Ly] + ez[(4*i+2)+j*Lx+k*Lx*Ly] * (Hz[(4*i+2)+j*Lx+k*Lx*Ly] - Hz[(4*i+2)+(j-1)*Lx+k*Lx*Ly] - Hy[(4*i+2)+j*Lx+k*Lx*Ly] + Hy[(4*i+2)+j*Lx+(k-1)*Lx*Ly] );
		    Ex[(4*i+3)+j*Lx+k*Lx*Ly] = Ex[(4*i+3)+j*Lx+k*Lx*Ly] + ez[(4*i+3)+j*Lx+k*Lx*Ly] * (Hz[(4*i+3)+j*Lx+k*Lx*Ly] - Hz[(4*i+3)+(j-1)*Lx+k*Lx*Ly] - Hy[(4*i+3)+j*Lx+k*Lx*Ly] + Hy[(4*i+3)+j*Lx+(k-1)*Lx*Ly] );
		    
		    Ey[4*i+j*Lx+k*Lx*Ly] = Ey[4*i+j*Lx+k*Lx*Ly] + ez[4*i+j*Lx+k*Lx*Ly] * (Hx[4*i+j*Lx+k*Lx*Ly] - Hx[4*i+j*Lx+(k-1)*Lx*Ly] - Hz[4*i+j*Lx+k*Lx*Ly] + Hz[4*i-1+j*Lx+k*Lx*Ly] );
		    Ey[(4*i+1)+j*Lx+k*Lx*Ly] = Ey[(4*i+1)+j*Lx+k*Lx*Ly] + ez[(4*i+1)+j*Lx+k*Lx*Ly] * (Hx[(4*i+1)+j*Lx+k*Lx*Ly] - Hx[(4*i+1)+j*Lx+(k-1)*Lx*Ly] - Hz[(4*i+1)+j*Lx+k*Lx*Ly] + Hz[(4*i)+j*Lx+k*Lx*Ly] );
		    Ey[(4*i+2)+j*Lx+k*Lx*Ly] = Ey[(4*i+2)+j*Lx+k*Lx*Ly] + ez[(4*i+2)+j*Lx+k*Lx*Ly] * (Hx[(4*i+2)+j*Lx+k*Lx*Ly] - Hx[(4*i+2)+j*Lx+(k-1)*Lx*Ly] - Hz[(4*i+2)+j*Lx+k*Lx*Ly] + Hz[(4*i+1)+j*Lx+k*Lx*Ly] );
		    Ey[(4*i+3)+j*Lx+k*Lx*Ly] = Ey[(4*i+3)+j*Lx+k*Lx*Ly] + ez[(4*i+3)+j*Lx+k*Lx*Ly] * (Hx[(4*i+3)+j*Lx+k*Lx*Ly] - Hx[(4*i+3)+j*Lx+(k-1)*Lx*Ly] - Hz[(4*i+3)+j*Lx+k*Lx*Ly] + Hz[(4*i+2)+j*Lx+k*Lx*Ly] );
		    
		    Ez[4*i+j*Lx+k*Lx*Ly] = Ez[4*i+j*Lx+k*Lx*Ly] + ez[4*i+j*Lx+k*Lx*Ly] * (Hy[4*i+j*Lx+k*Lx*Ly] - Hy[4*i-1+j*Lx+k*Lx*Ly] - Hx[4*i+j*Lx+k*Lx*Ly] + Hx[4*i+(j-1)*Lx+k*Lx*Ly] );
                    Ez[(4*i+1)+j*Lx+k*Lx*Ly] = Ez[(4*i+1)+j*Lx+k*Lx*Ly] + ez[(4*i+1)+j*Lx+k*Lx*Ly] * (Hy[(4*i+1)+j*Lx+k*Lx*Ly] - Hy[(4*i)+j*Lx+k*Lx*Ly] - Hx[(4*i+1)+j*Lx+k*Lx*Ly] + Hx[(4*i+1)+(j-1)*Lx+k*Lx*Ly] );
		    Ez[(4*i+2)+j*Lx+k*Lx*Ly] = Ez[(4*i+2)+j*Lx+k*Lx*Ly] + ez[(4*i+2)+j*Lx+k*Lx*Ly] * (Hy[(4*i+2)+j*Lx+k*Lx*Ly] - Hy[(4*i+1)+j*Lx+k*Lx*Ly] - Hx[(4*i+2)+j*Lx+k*Lx*Ly] + Hx[(4*i+2)+(j-1)*Lx+k*Lx*Ly] );
		    Ez[(4*i+3)+j*Lx+k*Lx*Ly] = Ez[(4*i+3)+j*Lx+k*Lx*Ly] + ez[(4*i+3)+j*Lx+k*Lx*Ly] * (Hy[(4*i+3)+j*Lx+k*Lx*Ly] - Hy[(4*i+2)+j*Lx+k*Lx*Ly] - Hx[(4*i+3)+j*Lx+k*Lx*Ly] + Hx[(4*i+3)+(j-1)*Lx+k*Lx*Ly] );
		    
		    Ez[Lx*(Ly/2)+Lx/2+(Lz/2)*Lx*Ly]=field;      ////// fonte
                    out[4*i+j*Lx+k*Lx*Ly]=Ez[4*i+j*Lx+k*Lx*Ly];
		    out[(4*i+1)+j*Lx+k*Lx*Ly]=Ez[(4*i+1)+j*Lx+k*Lx*Ly];
		    out[(4*i+2)+j*Lx+k*Lx*Ly]=Ez[(4*i+2)+j*Lx+k*Lx*Ly];
		    out[(4*i+3)+j*Lx+k*Lx*Ly]=Ez[(4*i+3)+j*Lx+k*Lx*Ly];
                }
      }

int main (int argc, char **argv)
{
  int i,j,k;  
/////////// Set Domain Sizes ////////////
  int Lx=32;													//computational x size set by user
  int Ly=32; 													//computational y size set by user
  int Lz=32;

dim3 dimBlock (BLOCKSIZEX, BLOCKSIZEY,BLOCKSIZEZ); 											//dimensions of threads block
 dim3 dimGrid (( Lx / (4*dimBlock.x) + ( Lx % (4*dimBlock.x) == 0?0:1)), ( Ly / (dimBlock.y) + ( Ly % (dimBlock.y) == 0?0:1)), ( Lz / (dimBlock.z) + ( Lz % (dimBlock.z) == 0?0:1))); 	//grid size that fits the user domain
   Lx=4*dimBlock.x*dimGrid.x;  												//computational x size 
  Ly=dimBlock.y*dimGrid.y; 												//computational y size 
  Lz=dimBlock.z*dimGrid.z;
  
 
  int D=Lx*Ly*Lz; 													//total computational domais. 
  int Dsize=D*sizeof(float);
     
  
//////////////////////////////////////////
// printf("%d %d\n",Lx,Ly);


///////////////////////////Physical Quantities/////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
float pi=4.0*atan(1.0);
float muo=4.0*pi*1.0e-7; // Permeability of free space
float epso=8.854e-12; // Permittivity of free space
float co=1.0/sqrt(muo*epso); // Speed of light in free space
//aimp=sqrt(muo/epso); // Wave impedance in free space
float dx=0.0001; // FDTD cell size
float dt=dx/co/sqrt(3.0); // Time step size

////////// eletrical permittivity ////////
float *ez_h;
ez_h = (float *)malloc(Dsize);
float *ez;
cudaMalloc ((void **) &ez, Dsize);



////////////////////////////////////////////
  for(i=0;i<Lx;i++)
    {
    for(j=0;j<Ly;j++)
        {
	for(k=0;k<Lz;k++)
	  {
	  ez_h[i+Lx*j+Lx*Ly*k]=dt/(epso*dx);
	  }
	}
    }
    
float mx = dt/muo/dx;
float my = mx;
///////////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////////
 ////////// Aloccate Ex field arrays //////// 
  float *Ex_h;
  Ex_h = (float *)malloc(Dsize);
  float *Ex;
  cudaMalloc ((void **) &Ex, Dsize);
  ////////// Aloccate Ey field arrays //////// 
  float *Ey_h;
  Ey_h = (float *)malloc(Dsize);
  float *Ey;
  cudaMalloc ((void **) &Ey, Dsize);
  ////////// Aloccate Ez field arrays //////// 
  float *Ez_h;
  Ez_h = (float *)malloc(Dsize);
  float *Ez;
  cudaMalloc ((void **) &Ez, Dsize);
  ////////// Aloccate Hxfield arrays //////// 
  float *Hx_h;
  Hx_h = (float *)malloc(Dsize);
  float *Hx;
  cudaMalloc ((void **) &Hx, Dsize);
  ////////// Aloccate Hy field arrays //////// 
  float *Hy_h;
  Hy_h = (float *)malloc(Dsize);
  float *Hy;
  cudaMalloc ((void **) &Hy, Dsize);
  ////////// Aloccate Hz field arrays //////// 
  float *Hz_h;
  Hz_h = (float *)malloc(Dsize);
  float *Hz;
  cudaMalloc ((void **) &Hz, Dsize);
  
  
    ////////// Debbuger out array //////// 
  float *out_h;
  out_h = (float *)malloc(Dsize);
  float *out;
  cudaMalloc ((void **) &out, Dsize);
  ////////////////////////////////////////////
  
  
/////////// Null Field initial condition //////
  

  for(i=0;i<Lx+1;i++)
    {
    for(j=0;j<Ly+1;j++)
        {
	for(k=0;k<Lz+1;k++)
	  { 
	  Ex_h[i+Lx*j+Lx*Ly*k]=0.f;
	  Ey_h[i+Lx*j+Lx*Ly*k]=0.f;
	  Ez_h[i+Lx*j+Lx*Ly*k]=0.f;
	  Hx_h[i+Lx*j+Lx*Ly*k]=0.f;
	  Hy_h[i+Lx*j+Lx*Ly*k]=0.f;
	  Hz_h[i+Lx*j+Lx*Ly*k]=0.f;
	  
	  }
        }
    }
   
     
     
///////////////////////////////////////////////////////////
/////////////// Coping data to Device /////////////////////
  cudaMemcpy (Ex, Ex_h, Dsize, cudaMemcpyHostToDevice);
  cudaMemcpy (Ey, Ey_h, Dsize, cudaMemcpyHostToDevice);
  cudaMemcpy (Ez, Ez_h, Dsize, cudaMemcpyHostToDevice);
  cudaMemcpy (Hx, Hx_h, Dsize, cudaMemcpyHostToDevice);
  cudaMemcpy (Hy, Hy_h, Dsize, cudaMemcpyHostToDevice);
  cudaMemcpy (Hz, Hz_h, Dsize, cudaMemcpyHostToDevice);
  cudaMemcpy (ez, ez_h, Dsize, cudaMemcpyHostToDevice);



///////////////////////////////////////////////////////////
////////////////////Time iteration ////////////////////////
  int T=4000;
  int b = 25.0;
  float dum,voltage,field;
  for (int t = 0; t < T; t = t + 1) //iterando no tempo
      {
	  dum = (4.0/b/dt)*(t*dt-b*dt);
	  voltage = 2.0*dum*exp(-(pow(dum,2.f)));
	// if(t<50)
	// {
	 field = voltage/dx;
	  
// 	 }
// 	 else
// 	 {
// 	  field=Ez_h[Lx*(Ly/2)+Lx/2+(Lz/2)*Lx*Ly];    
// 	
// 	 }
           WaveStepH <<< dimGrid, dimBlock >>> (Hx, Hy, Hz, Ex, Ey, Ez, Lx, Ly,Lz, mx, my,out);
           WaveStepE <<< dimGrid, dimBlock >>> (Hx, Hy, Hz, Ex, Ey, Ez, Lx, Ly,Lz, field, ez,out);
           checkCUDAError ("kernel invocation");
           cudaMemcpy (out_h, out, Dsize, cudaMemcpyDeviceToHost);
        // cudaMemcpy (Ez_h, Ez, Dsize, cudaMemcpyDeviceToHost);
        // cudaMemcpy (Hy_h, Hy, Dsize, cudaMemcpyDeviceToHost);
           checkCUDAError ("getting data from device");
          // dprint (Ez_h, Dx, Dy);
//         if (t%10==0)
//           {
//       	      cria_arquivo(out_h, Lx , Ly , t, Lx , Ly , Lz );
//            }
	 dprint ( out_h, Lx , Ly , Lx , Ly , Lz );
	   
     } 
      
     //    mprint ( Ez_h, Lx , Ly , Lx , Ly , Lz );

	   
}


                   
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
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

///////////////////////////////////////////////////////////
 void dprint (float *campo, int x, int y, int Lx, int Ly, int Lz)
    {
    for(int j = 0; j < y; j++)
      {
      for(int i = 0; i < x; i++)
	  {
// 	  if (campo[i+j*Lx+k*Lx*Ly]!=0.f)
	    {
	      printf("%d %d %f\n",i,j,campo[i+j*Lx+(Lz/2)*Lx*Ly]);
	//    printf("i=%d  j=%d  campo=%f\n",i,j,campo[i+j*Lx+(Lz/2)*Lx*Ly]);
	    
	    }
	    
	  }
      }
    }
    
///////////////////////////////////////////////////////////////////   
    void mprint (float *campo, int x, int y, int Lx, int Ly, int Lz )
    {
    for(int j = 0;j < y; j++)
      {
      for(int i = 0; i < x; i++)
	  {
	  printf("%g ", campo[i+j*Lx+(Lz/2)*Lx*Ly]);
	  }
      printf("\n");    
      }
    printf("\n");  
    }
 
////////////////////////////////////////////////////////////////////// 
    void midprint (float *campo, int Lx, int Ly, int Lz)
    {
    printf("%f ", campo[Lx*(Ly/2)+Lx/2+(Lz/2)*Lx*Ly]);
    printf("\n");  
    }

/////////////////////////////////////////////////////
void cria_arquivo(float *campo, int x, int y, int t, int Lx, int Ly, int Lz)
{
   FILE *onda;
   //remove("DATOS_ONDA");
   onda = fopen ("DATOS_ONDA", "a");
   fprintf(onda, "valor de t=%d ******************************************************************** \n",t);
     for(int j = 0; j < y; j++)
      {
      for(int i = 0; i < x; i++)
	 {
	   
	   fprintf(onda, "      i=%d  j=%d  campo=%f \n",i,j,campo[i+j*Lx+(Lz/2)*Lx*Ly]);
	    
	 }
      }
   
   fclose(onda);
   
    
}
      
 


    
    
