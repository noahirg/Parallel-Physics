#pragma once
#include <vector>
#include <memory>
#include "Circle.hpp"

const int ITER = 8;

class PhyWorld
{
    public:
    bool checkCollisions;
    std::vector<Circle> bodies;

    PhyWorld(bool check = false);
    void update(float dt);
    void solveCollisions();
    void applyConstraint();
    void updatePositions(float dt);
    Circle* createCircle(Vec2f pos, float mass, float rad, bool pinned = false);
};