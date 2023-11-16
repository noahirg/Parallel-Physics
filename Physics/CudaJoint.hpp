#pragma once

#include <cmath>
#include "CudaCircle.hpp"

class CudaJoint
{
    public:

    CudaJoint(float length, int inter1, int inter2) 
    : length(length), cir1(inter1), cir2(inter2)
    {}


    void
    update(CudaCircle* cir1, CudaCircle* cir2, float dt) 
    {
        float axisx = cir1->posx - cir2->posx;
        float axisy = cir1->posy - cir2->posy;
        float dist = std::sqrt(axisx * axisx + axisy * axisy);
        float nx = axisx / dist;
        float ny = axisy / dist;
        float delta = dist - length;
        //delta *= .1f;
        /*if (!cir1->pinned)
            cir1->pos += .5f * delta * n;
        if (!cir2->pinned)
            cir2->pos -= .5f * delta * n;
        */
        //Relative velocity
        float relx = (cir1->posx - cir1->posOldx) - (cir2->posx - cir2->posOldx);
        float rely = (cir1->posy - cir1->posOldy) - (cir2->posy - cir2->posOldy);
        float dampenx = (relx / dt) * 1.f;
        float dampeny = (rely / dt) * 1.f;
        float k = 20000.f;
        float forcex = -k * delta * nx;
        float forcey = -k * delta * ny;
        forcex -= dampenx;
        forcey -= dampeny;
        //prevForce = rel;
        //Using Hooke's law
        cir1->applyForce (forcex, forcey);
        cir2->applyForce (-forcex, -forcey);
    }

    float length;
    float curLength;
    int cir1;
    int cir2;
    float prevForcex;
    float prevForcey;
    float dampen;
};