#pragma once
#include "CudaGrid.cuh"


/*class CudaCell
{
    public:
    CudaCell(int x, int y, int width, int height, CudaCircle* cir, CudaCell* cellsL)
        : m_x(x), m_y(y), m_width(width), m_height(height), cir(cir), cells(cellsL)
    {}

    CudaCell()
        : m_x(0), m_y(0), m_width(0), m_height(0)
    {}

    void
    add(float x, float y, unsigned id)
    {
        m_ids.push_back(id);
    }

    void
    update()
    {
        //No updating if no elements
        if (m_ids.size() == 0)
            return;
        std::vector<int> tempRem;
        //Make sure elements still in this cell
        for (unsigned i = 0; i < m_ids.size(); ++i)
        {
            //Need to account for deleted circles
            //Can add id to circle class and check if it exists still


            //If not in cell remove
            if (!inCell(cir[m_ids[i]].posx, cir[m_ids[i]].posy))
            {
                tempRem.push_back(m_ids[i]);
            }
        }
        //Remove points not in cell
        for (unsigned i = 0; i < tempRem.size(); ++i)
        {
            m_ids.erase(std::remove(m_ids.begin(), m_ids.end(), tempRem[i]), m_ids.end());
            //Add them back into Grid
            int indx = static_cast<int>(cir[tempRem[i]].posx) / m_width;
            int indy = static_cast<int>(cir[tempRem[i]].posy) / m_height;
            cells[indx + indy * DIV].add(x, y, tempRem[i]);

            //phy->insertToGrid(cir[tempRem[i]].posx, cir[tempRem[i]].posy, tempRem[i]);
        }
    }

    bool
    intersects(float x, float y, float rad)
    {
        //find if circle intersects rectangle of Quad
        float Xn = std::max(static_cast<float>(m_x), std::min(x, static_cast<float>(m_x + m_width)));
        float Yn = std::max(static_cast<float>(m_y), std::min(y, static_cast<float>(m_y + m_height)));
        
        float Dx = Xn - x;
        float Dy = Yn - y;
        return (Dx * Dx + Dy * Dy) <= rad * rad;
    }

    bool
    rectIntersects(float x, float y, float w, float h)
    {
        return x < static_cast<float>(m_x + m_width) && (x + w) > static_cast<float>(m_x) &&
            y < static_cast<float>(m_y + m_height) && (y + h) > static_cast<float>(m_y);
    }

    bool
    inCell(float x, float y)
    {
        return (x >= m_x && x < (m_x + m_width) && y >= m_y && y < (m_y + m_height));
    }

    std::vector<unsigned>
    query(float x, float y, float rad)
    {
        //Range isn't in cell
        if (!intersects(x, y, rad))
            return {};

        //Range in cell
        return m_ids;
    }

    /*
    std::array<int, 4>
    getCell()
    {   
        return {m_x, m_y, m_width, m_height};
    }*-/

    
    int m_x;
    int m_y;
    int m_width;
    int m_height;
    CudaCircle* cir;
    CudaCell* cells;
    std::vector<unsigned> m_ids {};
};
*/


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
    //cudaMemcpy(cudaCells, m_cells, DIV * DIV * sizeof(cudaCells), cudaMemcpyHostToDevice);


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
    
    //m_cells[indx + indy * DIV].add(x, y, id);
}

/*void
initAdd()
{
    for (unsigned i = 0; i < phy->numEle; ++i)
    {
        addSingle(phy->cir[i].posx, phy->cir[i].posy, i);
    }
}*/

void
CudaGrid::update(unsigned numEle)
{
    /*for (unsigned i = 0; i < m_cells.size(); ++i){
        m_cells[i].update();}*/

    /*for (unsigned j = 0; j < DIV; ++j)
    {
        for (unsigned i = 0; i < DIV; ++i)
        {
            
            //int ind = i + j * DIV + DIV + 3 + (2 * j);
            m_cells[i + j * DIV].update();
            
        }
    }*/
    //Set all cell counts to 0
    int blockSize = 256;
    int numBlocks = (DIV * DIV + blockSize - 1) / blockSize;


    resetCounts<<<numBlocks, blockSize>>>(cudaCells);
    cudaDeviceSynchronize();

    //Call addToGrid to re-add all particles
    numBlocks = (numEle + blockSize - 1) / blockSize;
    addToGrid<<<numBlocks, blockSize>>>(cir, cudaCells, cellW, cellH, numEle);
    //addToGrid<<<1, 1>>>(cir, cudaCells, cellW, cellH, numEle);
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
    /*for (unsigned i = 0; i < numEle; ++i)
    {
        int indx = static_cast<int>(cir[i].posx) / cellW;
        int indy = static_cast<int>(cir[i].posy) / cellH;
        
        CudaCell& curCell = cells[indx + indy * DIV];
        
        //int index = atomicAdd(&curCell.count, 1);
        ++curCell.count;
        curCell.m_ids[curCell.count - 1] = i;
    }*/
    int idx = threadIdx.x + blockIdx.x * blockDim.x;


    if (idx >= numEle)
        return;


    int indx = static_cast<int>(cir[idx].posx) / cellW;
    int indy = static_cast<int>(cir[idx].posy) / cellH;
    
    int index = atomicAdd(&(cells[indx + indy * DIV].count), 1);
    
    //++cells[indx + indy * DIV].count;
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