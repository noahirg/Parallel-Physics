#pragma once
#include <vector>
#include <memory>
#include "Circle.hpp"
#include "Joint.hpp"

const int ITER = 8;

//:public PhyGen
class PhyWorld
{
    public:
    bool checkCollisions;
    std::vector<Circle> bodies;
    std::vector<Joint> joints;
    Vec2f worldSize;


    PhyWorld(int sizeX, int sizeY, bool check = false);
    virtual void update(float dt);
    virtual void solveCollisions();
    virtual void applyConstraint();
    virtual void updateJoints(float dt);
    virtual void updatePositions(float dt);
    virtual Circle* createCircle(Vec2f pos, float mass, float rad, bool pinned = false);
    virtual Joint* createJoint(float length,int cir1, int cir2);
    virtual void insertToGrid(Vec2f pos, unsigned id);
};