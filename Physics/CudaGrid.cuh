#pragma once
#include <vector>
#include <algorithm>
#include "CudaCircle.cuh"

const int DIV = 100;


class CudaCell
{
    public:

    CudaCell();

    void
    add(float x, float y, unsigned id);
    
    unsigned count;
    unsigned m_ids[20];
};

class CudaGrid
{
    public:
    CudaGrid();

    CudaGrid(int width, int height, CudaCircle* cir, unsigned numEle);

    ~CudaGrid();


    void
    addSingle(float x, float y, unsigned id);



    void
    update(unsigned numEle);


    int
    getCellCount();




    CudaCircle* cir;
    int m_width;
    int m_height;
    int cellH;
    int cellW;
    CudaCell* cudaCells;
};
