#pragma once
#include <vector>
#include <memory>
#include "CudaGrid.cuh"
#include "PhyCWorld.cuh"
#include "CudaCircle.cuh"
#include "CudaJoint.hpp"

const unsigned MAX_CIR_CU = 5'000'000;

class PhyCuda : public PhyCWorld
{
    public:

    PhyCuda(int sizeX, int sizeY, bool check = false);// : PhyWorld(sizeX, sizeY, check);
    /*{
        grid = new Grid(static_cast<int>(worldSize.x), static_cast<int>(worldSize.y), this);
    }*/

    ~PhyCuda();
    /*{
        delete grid;
    }*/

    void
    update(float dt);
    /*{
        for (int k = 0; k < ITER; ++k)
        {
            tempColor();
            splitCells();
            updateJoints(dt / static_cast<float>(ITER));
            updatePositions(dt / static_cast<float>(ITER));
            applyConstraint();
            grid->update();
        }
    }*/

    void
    tempColor();
    /*
        for (int i = 0; i < bodies.size(); ++i)
        {
            bodies[i].red = 255;
            bodies[i].blue = 255;
            bodies[i].green = 255;
        }
    }*/

    //Number of cells must be divisible by 2 * threadCount for this to work
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


    CudaCircle* 
    createCircle(float posx, float posy, float mass, float rad, bool pinned = false);
    /*{
        bodies.emplace_back( pos, mass, rad, pinned );
        //insertToGrid(pos, bodies.size() - 1);
        grid->addSingle(pos.x, pos.y, bodies.size() - 1);
        //tree->addSingle(pos.x, pos.y, bodies.size() - 1);
        return &bodies.back();
    }*/

    void
    insertToGrid(float posx, float posy, unsigned id);
    /*{
        grid->addSingle(pos.x, pos.y, id);
    }*/

    std::vector<std::array<int, 4>>
    getGrid();
    /*{
        return grid->getCells();
    }*/

    CudaGrid* grid;
};

