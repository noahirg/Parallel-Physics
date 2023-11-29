#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include "CudaCircle.cuh"
#include "CudaJoint.hpp"


class PhyCWorld
{
    public:
    bool checkCollisions;
    //std::vector<CudaCircle> bodies;
    CudaCircle* bodies;
    std::vector<CudaJoint> joints;
    float worldSizex;
    float worldSizey;
    unsigned* ids;
    unsigned* idLoc;
    CudaCircle* cir;
    unsigned numEle;


    PhyCWorld(int sizeX, int sizeY, bool check = false);
    virtual void update(float dt);
    virtual void applyConstraint();
    virtual void updateJoints(float dt);
    virtual void updatePositions(float dt);
    virtual CudaCircle* createCircle(float posx, float posy, float mass, float rad, bool pinned = false);
    virtual CudaJoint* createJoint(float length,int cir1, int cir2);
    virtual void insertToGrid(float posx, float posy, unsigned id);
};