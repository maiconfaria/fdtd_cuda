#define BLOCKSIZEX 16
#define BLOCKSIZEY 16

#ifndef _INITPARAM_H_
#define _INITPARAM_H_


  

     float pi=4.0*atan(1.0);
     float muo=4.0*pi*1.0e-7;       // Permeability of free space
     float epso=8.854e-12;          // Permittivity of free space
     float co=1.0/sqrt(muo*epso);   // Speed of light in free space
     float eta=sqrt(muo/epso);     // Wave impedance in free space
     float dx=0.000001;               // FDTD cell size
     float dy=0.000001;               // FDTD cell size
     float dt=dx/co/sqrt(2.0);            // Time step size

     int pml_lx = 16;												//pml x lenght
     int pml_ly = 16;												//pml y lenght
     
     int lx=512;													//computational x size set by user
     int ly=512; 													//computational y size set by user
    
  
     dim3 dimBlock (BLOCKSIZEX, BLOCKSIZEY); 											//dimensions of threads block
     dim3 dimGrid (( (lx + 2 * pml_lx) / (dimBlock.x) + ( (lx + 2 * pml_lx) % (dimBlock.x) == 0?0:1)), ( (ly + 2 * pml_ly) / (dimBlock.y) + ( (ly + 2 * pml_ly) % (dimBlock.y) == 0?0:1))); 	//grid size that fits the user domain
  
     int Lx=dimBlock.x*dimGrid.x; 												//computational x size 
     int Ly=dimBlock.y*dimGrid.y; 												//computational y size 
     int D=Lx*Ly; 													//total computational domais. 
     int Dsize=D*sizeof(float);
     
     
#endif