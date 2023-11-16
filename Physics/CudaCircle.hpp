#pragma once
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