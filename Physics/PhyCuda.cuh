#pragma once
#include <vector>
#include <memory>
#include "CudaGrid.cuh"
#include "CudaCircle.cuh"
#include "CudaJoint.hpp"

const unsigned MAX_CIR_CU = 5'000'000;

class PhyCuda
{
    public:

    PhyCuda(int sizeX, int sizeY, bool check = false);

    ~PhyCuda();

    void
    update(float dt);

    void
    tempColor();

    void
    splitCells();
    

    void
    updateJoints(float dt);

    void
    updatePositions(float dt);

    void
    applyConstraint();

    void
    applyForceAll(float fx, float fy);


    void 
    createCircle(float posx, float posy, float mass, float rad, bool pinned = false);

    void
    insertToGrid(float posx, float posy, unsigned id);

    std::vector<std::array<int, 4>>
    getGrid();

    CudaGrid* grid;
    float gravity {};
    bool checkCollisions;
    CudaCircle* bodies;
    std::vector<CudaJoint> joints;
    float worldSizex;
    float worldSizey;
    unsigned* ids;
    unsigned* idLoc;
    CudaCircle* cir;
    unsigned numEle;
};

