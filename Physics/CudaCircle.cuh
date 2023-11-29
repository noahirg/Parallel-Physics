#pragma once
#include <cuda_runtime.h>
#include <iostream>

class CudaCircle
{
    public:
    float posx {};
    float posy {};
    float accx {};
    float accy {};
    bool pinned {};
    float posOldx {};
    float posOldy {};
    float mass {};
    float rad {};

    CudaCircle() = default;

    CudaCircle(float x, float y, float mass, float rad, bool pinned = false) : posx(x), posy(y), mass(mass), rad(rad), pinned(pinned), posOldx(x), posOldy(y)
    {}

    __device__
    void
    update(float dt)
    {
        if (pinned) return;
        float velox = posx - posOldx;
        float veloy = posy - posOldy;

        posOldx = posx;
        posOldy = posy;
        posx += velox + accx * dt * dt;
        posy += veloy + accy * dt * dt;

        accx = {};
        accy = {};
    }

    __device__
    void
    applyForce(float fx, float fy)
    {
        if (pinned) return;
        if (mass == 0) 
        {
            accx = {};
            accy = {};
        }
        else 
        {
            accx += fx / mass;
            accy += fy / mass;
        }
    }

    void
    applyAcc(float accxT, float accyT)
    {
        if (pinned) return;
        accx += accxT;
        accy += accyT;
    }

    void
    pin()
    {
        pinned = true;
    }

    float
    getRad(float otherPosx, float otherPosy)
    {
        return rad;
    }

    
};

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}