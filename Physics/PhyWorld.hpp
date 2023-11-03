#pragma once
#include <vector>
#include <memory>
#include "Circle.hpp"

const int ITER = 8;

//:public PhyGen
class PhyWorld
{
    public:
    bool checkCollisions;
    std::vector<Circle> bodies;
    Vec2f worldSize;


    PhyWorld(int sizeX, int sizeY, bool check = false);
    void update(float dt);
    void solveCollisions();
    void applyConstraint();
    void updatePositions(float dt);
    Circle* createCircle(Vec2f pos, float mass, float rad, bool pinned = false);
    void insertToTree(Vec2f pos, unsigned id);
};