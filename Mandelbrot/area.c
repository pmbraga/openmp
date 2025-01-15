#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

struct complex{
  double real;
  double imag;
};

int main(){
  int numoutside = 0;
  int NPOINTS = 2000;
  int MAXITER = 1000;
  double area, error;
  double start, finish;
  struct complex z, c;

/* Outer loops run over npoints, initialise z=c. Inner loop has the iteration z=z*z+c, and threshold test
 for (int i= high * myid ; i< high * (myid+1); i++) {*/

  start = omp_get_wtime();
    
#pragma omp parallel default(none) private(z,c) shared(NPOINTS, MAXITER) reduction(+:numoutside)
    {
        
        int no_threads = omp_get_num_threads();
        printf("No. of threads: %12.8d\n",no_threads);
        int myid = omp_get_thread_num();
        int interval = NPOINTS/no_threads;
    
        
        for (int i= interval * myid ; i< interval * (myid + 1); i++) {
            
            for (int j=0; j<NPOINTS; j++) {
                
                c.real = -2.0+2.5*(double)(i)/(double)(NPOINTS)+1.0e-7;
                c.imag = 1.125*(double)(j)/(double)(NPOINTS)+1.0e-7;
                z=c;
                
                for (int iter=0; iter<MAXITER; iter++){
                    
                    double ztemp=(z.real*z.real)-(z.imag*z.imag)+c.real;
                    z.imag=z.real*z.imag*2+c.imag;
                    z.real=ztemp;
                    
                    if ((z.real*z.real+z.imag*z.imag)>4.0e0) {
                        numoutside++;
                        break;
                    }
                }
            }
        }
        
    }


  finish = omp_get_wtime();  

/* Calculate area and error and output the results */

      area=2.0*2.5*1.125*(double)(NPOINTS*NPOINTS-numoutside)/(double)(NPOINTS*NPOINTS);
      error=area/(double)NPOINTS;

      printf("Area of Mandlebrot set = %12.8f +/- %12.8f\n",area,error);
      printf("Time = %12.8f seconds\n",finish-start);

  }


/*
 * BASICS: Monte Carlo approach, similar to calculating pi using random number generation
 * GOAL: Parallelise the testing if a point is within the set or not
 * Each thread does 2000 iteration in a specific section of space,
 *
 */
