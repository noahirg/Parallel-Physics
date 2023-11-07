#pragma once
#include "libr/utils.hpp"

class Body
{
    public:
    Vec2f pos;
    Vec2f acc;
    bool pinned;
    Vec2f posOld;
    float mass;
    float rad;
    short red = 255;
    short blue = 255;
    short green = 255;

    Body();
    Body(float x, float y, float mass, bool pinned = false);
    Body(Vec2f pos, float mass, bool pinned = false);
    void update(float dt);
    void applyForce(Vec2f force);
    void applyForce(float fx, float fy);
    void applyAcc(Vec2f acc);
    void applyAcc(float accx, float accy);
    void pin();
    virtual float getRad(Vec2f otherPos) {return rad;}
    std::tuple<short, short, short> getColor();
};