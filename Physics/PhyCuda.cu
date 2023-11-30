#include "PhyCuda.cuh"


__global__
void
solveEle(CudaCircle* bodies, CudaCell* cells, unsigned numEle, int cellW, int cellH, const int DIV);

__global__
void
solveCell(CudaCircle* bodies, CudaCell* cells, const int DIV);

__device__
void
checkEleCol(unsigned id, unsigned idx, CudaCircle* bodies, CudaCell* cells);

__device__
void
solveCollision(unsigned i, unsigned j, CudaCircle* bodies);

__global__
void
updatePos(unsigned count, CudaCircle* cir, float dt, float gravity);

__global__
void
applyCon(unsigned count, CudaCircle* cir, float worldSizex, float worldSizey);

__global__
void
applyForAll(unsigned count, CudaCircle* cir, float fx, float fy);


PhyCuda::PhyCuda(int sizeX, int sizeY, bool check) : checkCollisions(check), worldSizex(sizeX), worldSizey(sizeY)
{
    bodies = new CudaCircle[5'000'000];
    cudaMalloc(&cir, MAX_CIR_CU * sizeof(CudaCircle));
    numEle = 0;
    grid = new CudaGrid(static_cast<int>(worldSizex), static_cast<int>(worldSizey), cir, numEle);

}

PhyCuda::~PhyCuda()
{
    delete grid;
    cudaFree(cir);
}

void
PhyCuda::update(float dt)
{
    const int ITERC = 8;


    for (int k = 0; k < ITERC; ++k)
    {
        splitCells();
        //updateJoints(dt / static_cast<float>(ITERC));
        updatePositions(dt / static_cast<float>(ITERC));
        applyConstraint();
        grid->update(numEle);
    }

    //For rendering
    cudaMemcpy(bodies, cir, numEle * sizeof(CudaCircle), cudaMemcpyDeviceToHost);
}

void
PhyCuda::updateJoints(float dt)
{
    for (unsigned i = 0; i < joints.size(); ++i)
    {
        joints[i].update(&cir[joints[i].cir1], &cir[joints[i].cir2], dt);
    }
}

void
PhyCuda::updatePositions(float dt)
{
    //GPUd
    int blockSize = 256;
    int numBlocks = (numEle + blockSize - 1) / blockSize;
    updatePos<<<numBlocks, blockSize>>>(numEle, cir, dt, gravity);

    cudaDeviceSynchronize();
}

__global__
void
updatePos(unsigned count, CudaCircle* cir, float dt, float gravity)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= count)
        return;

    cir[idx].accy += gravity;
    cir[idx].update(dt);
}

void
PhyCuda::applyConstraint()
{
    int blockSize = 256;
    int numBlocks = (numEle + blockSize - 1) / blockSize;
    applyCon<<<numBlocks, blockSize>>>(numEle, cir, worldSizex, worldSizey);

    cudaDeviceSynchronize();
}

__global__
void
applyCon(unsigned count, CudaCircle* cir, float worldSizex, float worldSizey)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= count)
        return;


    if (cir[idx].posx > worldSizex - cir[idx].rad)
        cir[idx].posx = worldSizex - cir[idx].rad;
    else if (cir[idx].posx < cir[idx].rad)
        cir[idx].posx = cir[idx].rad;

    if (cir[idx].posy > worldSizey - cir[idx].rad)
        cir[idx].posy = worldSizey - cir[idx].rad;
    else if (cir[idx].posy < cir[idx].rad)
        cir[idx].posy = cir[idx].rad;
}

void
PhyCuda::applyForceAll(float fx, float fy)
{
    int blockSize = 256;
    int numBlocks = (numEle + blockSize - 1) / blockSize;
    applyForAll<<<numBlocks, blockSize>>>(numEle, cir, fx, fy);

    cudaDeviceSynchronize();
}

__global__
void
applyForAll(unsigned count, CudaCircle* cir, float fx, float fy)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= count)
        return;

    cir[idx].applyForce(fx, fy);
}

void
PhyCuda::splitCells()
{
    int cellPerPass = DIV * DIV;
    int blockSize = 256;
    int numBlocks = (cellPerPass + blockSize - 1) / blockSize;
    
    numBlocks = (numEle + blockSize - 1) / blockSize;
    solveEle<<<numBlocks, blockSize>>>(cir, grid->cudaCells, numEle, grid->cellW, grid->cellH, DIV);
    cudaDeviceSynchronize();
}

__global__
void
solveEle(CudaCircle* bodies, CudaCell* cells, unsigned numEle, int cellW, int cellH, const int DIV)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id >= numEle)
        return;

    //Calculate cell
    int indx = static_cast<int>(bodies[id].posx) / cellW;
    int indy = static_cast<int>(bodies[id].posy) / cellH;
    int idx = indx + indy * DIV;
    
    if (idx / DIV != 0 && idx % DIV != 0)
        checkEleCol(id, idx - DIV - 1, bodies, cells);
        
    if (idx / DIV != 0)
        checkEleCol(id, idx - DIV, bodies, cells);

    if (idx / DIV != 0 && (idx + 1) % DIV != 0) 
        checkEleCol(id, idx - DIV + 1, bodies, cells);

    if (idx % DIV != 0)
        checkEleCol(id, idx - 1, bodies, cells);

    checkEleCol(id, idx, bodies, cells);

    if ((idx + 1) % DIV != 0)
        checkEleCol(id, idx + 1, bodies, cells);

    if (idx / DIV != DIV - 1 && idx % DIV != 0)
        checkEleCol(id, idx + DIV - 1, bodies, cells);

    if (idx / DIV != DIV - 1)
        checkEleCol(id, idx + DIV, bodies, cells);

    if (idx / DIV != DIV - 1 && (idx + 1) % DIV != 0)
        checkEleCol(id, idx + DIV + 1, bodies, cells);
}

__global__
void
solveCell(CudaCircle* bodies, CudaCell* cells, const int DIV)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= DIV * DIV)
        return;


    int cellSize = cells[idx].count;

    if (cellSize == 0)
        return;


    for (int i = 0; i < cellSize; ++i)
    {
        unsigned id = cells[idx].m_ids[i];
        
        if (idx / DIV != 0 && idx % DIV != 0)
            checkEleCol(id, idx - DIV - 1, bodies, cells);
            
        if (idx / DIV != 0)
            checkEleCol(id, idx - DIV, bodies, cells);

        if (idx / DIV != 0 && (idx + 1) % DIV != 0) 
            checkEleCol(id, idx - DIV + 1, bodies, cells);

        if (idx % DIV != 0)
            checkEleCol(id, idx - 1, bodies, cells);

        checkEleCol(id, idx, bodies, cells);

        if ((idx + 1) % DIV != 0)
            checkEleCol(id, idx + 1, bodies, cells);

        if (idx / DIV != DIV - 1 && idx % DIV != 0)
            checkEleCol(id, idx + DIV - 1, bodies, cells);

        if (idx / DIV != DIV - 1)
            checkEleCol(id, idx + DIV, bodies, cells);

        if (idx / DIV != DIV - 1 && (idx + 1) % DIV != 0)
            checkEleCol(id, idx + DIV + 1, bodies, cells);
    }
}


__device__
void
checkEleCol(unsigned id, unsigned idx, CudaCircle* bodies, CudaCell* cells)
{
    int cellSize = cells[idx].count;
    
    for (int i = 0; i < cellSize; ++i)
    {
        solveCollision(id, cells[idx].m_ids[i], bodies);
    }
}

__device__
void
solveCollision(unsigned i, unsigned j, CudaCircle* bodies)
{
    float epsilon = .0001f;

    if (i == j)
        return;


    float colAxisx = bodies[i].posx - bodies[j].posx;
    float colAxisy = bodies[i].posy - bodies[j].posy;
    float distSq = colAxisx * colAxisx + colAxisy * colAxisy;
    // push the shape along the normal of that line
    float iRad = bodies[i].rad;
    float jRad = bodies[j].rad;

    float radD = iRad + jRad;
    if (distSq < radD * radD && distSq > epsilon)
    {
        float dist = sqrtf(distSq);
        float normalx = colAxisx / dist;
        float normaly = colAxisy / dist;
        float delta = radD - dist;
        float di = (jRad / radD) * delta;
        float dj = (iRad / radD) * delta;

        if (bodies[i].pinned && bodies[j].pinned)
            {di = 0; dj = 0;}
        else if (bodies[i].pinned)
            dj = delta;
        else if (bodies[j].pinned)
            di = delta;

        bodies[i].posx += di * normalx;
        bodies[i].posy += di * normaly;
        bodies[j].posx -= dj * normalx;
        bodies[j].posy -= dj * normaly;
    }
}

void 
PhyCuda::createCircle(float posx, float posy, float mass, float rad, bool pinned)
{
    CudaCircle circ (posx, posy, mass, rad, pinned);
    if (pinned)
    {
        circ.accx = 1000000.f;
        circ.accy = 100000.f;
    }
    circ.pinned = false;
    cudaMemcpy(&(cir[numEle]), &circ, sizeof(CudaCircle), cudaMemcpyHostToDevice);
    ++numEle;
}

void
PhyCuda::insertToGrid(float posx, float posy, unsigned id)
{
    grid->addSingle(posx, posy, id);
}