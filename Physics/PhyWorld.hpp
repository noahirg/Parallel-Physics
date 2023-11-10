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
    void update(float dt);
    void solveCollisions();
    void applyConstraint();
    void updateJoints();
    void updatePositions(float dt);
    virtual Circle* createCircle(Vec2f pos, float mass, float rad, bool pinned = false);
    Joint* createJoint(float length,int cir1, int cir2);
    virtual void insertToGrid(Vec2f pos, unsigned id);
};