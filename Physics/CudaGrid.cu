#pragma once
#include "CudaGrid.cuh"


CudaCell::CudaCell() : count(0)
{}

void
CudaCell::add(float x, float y, unsigned id)
{
    m_ids[count] = id;
    ++count;
}



__global__
void
addToGrid(CudaCircle* cir, CudaCell* cells, int cellW, int cellH, int numEle);

__global__
void
resetCounts(CudaCell* cells);


CudaGrid::CudaGrid()
{}

CudaGrid::CudaGrid(int width, int height, CudaCircle* cir, unsigned numEle) : m_width(width), m_height(height), cellW(width / DIV),  cellH(height / DIV), cir(cir)
{
    //Divide world into cells
    //Make sure width and height divisible by DIV

    cudaMalloc(&cudaCells, DIV * DIV * sizeof(CudaCell));


    //Use GPU to put objects into grid
    int blockSize = 256;
    int numBlocks = (numEle + blockSize - 1) / blockSize;
    addToGrid<<<numBlocks, blockSize>>>(cir, cudaCells, cellW, cellH, numEle);
    cudaDeviceSynchronize();
}

CudaGrid::~CudaGrid()
{
    cudaFree(cudaCells);
}


void
CudaGrid::addSingle(float x, float y, unsigned id)
{
    //Check to which cell it goes in
    int indx = static_cast<int>(x) / cellW;
    int indy = static_cast<int>(y) / cellH;
    
}


void
CudaGrid::update(unsigned numEle)
{

    //Set all cell counts to 0
    int blockSize = 256;
    int numBlocks = (DIV * DIV + blockSize - 1) / blockSize;


    resetCounts<<<numBlocks, blockSize>>>(cudaCells);
    cudaDeviceSynchronize();

    //Call addToGrid to re-add all particles
    numBlocks = (numEle + blockSize - 1) / blockSize;
    addToGrid<<<numBlocks, blockSize>>>(cir, cudaCells, cellW, cellH, numEle);
    cudaDeviceSynchronize();
}

int
CudaGrid::getCellCount()
{
    return DIV * DIV;
}

__global__
void
addToGrid(CudaCircle* cir, CudaCell* cells, int cellW, int cellH, int numEle)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;


    if (idx >= numEle)
        return;


    int indx = static_cast<int>(cir[idx].posx) / cellW;
    int indy = static_cast<int>(cir[idx].posy) / cellH;
    
    int index = atomicAdd(&(cells[indx + indy * DIV].count), 1);
    

    cells[indx + indy * DIV].m_ids[index] = idx;
}

__global__
void
resetCounts(CudaCell* cells)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= DIV * DIV)
        return;

    cells[idx].count = 0;
}