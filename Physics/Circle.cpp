#include "Circle.hpp"

Circle::Circle(float x, float y, float mass, float rad, bool pinned) : rad(rad), Body(Vec2f(x, y), mass, pinned)
{}

Circle::Circle(Vec2f pos, float mass, float rad, bool pinned) : rad(rad), Body(pos, mass, pinned)
{}

float
Circle::getRad(Vec2f otherPos)
{
    return rad;
}