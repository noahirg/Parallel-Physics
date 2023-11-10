#pragma once
#include "Body.hpp"

class Circle : public Body
{
    public:
    float rad;

    Circle() {};
    Circle(float x, float y, float mass, float rad, bool pinned = false);
    Circle(Vec2f pos, float mass, float rad, bool pinned = false);
    float getRad(Vec2f otherPos);
};