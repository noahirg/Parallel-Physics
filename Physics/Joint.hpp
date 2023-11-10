#pragma once

#include "Circle.hpp"

class Joint
{
    public:

    Joint(float length, int inter1, int inter2) 
    : length(length), cir1(inter1), cir2(inter2)
    {}


    void
    update(Circle* cir1, Circle* cir2) 
    {
        Vec2f axis = cir1->pos - cir2->pos;
        float dist = axis.magn();
        Vec2f n = axis / dist;
        float delta = dist - length;
        //delta *= .1f;
        /*if (!cir1->pinned)
            cir1->pos += .5f * delta * n;
        if (!cir2->pinned)
            cir2->pos -= .5f * delta * n;
        */
        float k = 1000.f;
        float dampen = .7f;
        Vec2f force = -k * delta * n * dampen;
        //Using Hooke's law
        cir1->applyForce (force);
        cir2->applyForce (-force);
    }

    float length;
    float curLength;
    int cir1;
    int cir2;
    Vec2f prevForce;
    float dampen;
};