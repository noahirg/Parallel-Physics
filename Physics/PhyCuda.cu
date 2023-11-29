#include "PhyCuda.cuh"


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
updatePos(unsigned count, CudaCircle* cir, float dt);

__global__
void
applyCon(unsigned count, CudaCircle* cir, float worldSizex, float worldSizey);

__global__
void
applyForAll(unsigned count, CudaCircle* cir, float fx, float fy);


PhyCuda::PhyCuda(int sizeX, int sizeY, bool check) : PhyCWorld(sizeX, sizeY, check)
{
    /*ids = new unsigned[MAX_CIR_CU + (DIV * DIV)];
    idLoc = new unsigned[DIV * DIV]();
    cudaMalloc(&ids, MAX_CIR_CU + (DIV * DIV));
    cudaMalloc(&idLoc, DIV * DIV);*/
    cudaMalloc(&cir, MAX_CIR_CU * sizeof(CudaCircle));
    numEle = 0;
    grid = new CudaGrid(static_cast<int>(worldSizex), static_cast<int>(worldSizey), cir, numEle);


    /**
     * need to find alternative to cudamalloc managed
     * essentially all data needs to live on the gpu during ITERC iterations in update loop
     * then data should be transfered back to cpu so it can be rendered
     * 
     */
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

    //cudaMemcpy(cir, &bodies[0], bodies.size() * sizeof(CudaCircle), cudaMemcpyHostToDevice);

    for (int k = 0; k < ITERC; ++k)
    {
        //tempColor();
        splitCells();
        /*if (numEle > 0)
            std::cout << "piss1" << std::endl;*/
        //updateJoints(dt / static_cast<float>(ITERC));
        updatePositions(dt / static_cast<float>(ITERC));
        applyConstraint();
        grid->update(numEle);
    }

    //For render purposes
    cudaMemcpy(bodies, cir, numEle * sizeof(CudaCircle), cudaMemcpyDeviceToHost);
    /*if (numEle > 0)
            std::cout << "posx: " << bodies[0].posx << "    posy: " << bodies[0].posy << std::endl;*/
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
    updatePos<<<numBlocks, blockSize>>>(numEle, cir, dt);

    cudaDeviceSynchronize();
}

__global__
void
updatePos(unsigned count, CudaCircle* cir, float dt)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= count)
        return;

    /*if (idx == 0)
            printf("valx: %f\n", cir[0].posx);*/

    cir[idx].update(dt);
}

void
PhyCuda::applyConstraint()
{
    int blockSize = 256;
    int numBlocks = (numEle + blockSize - 1) / blockSize;
    applyCon<<<numBlocks, blockSize>>>(numEle, cir, worldSizex, worldSizey);

    cudaDeviceSynchronize();
    /*for (int i = 0; i < numEle; ++i)
    {
        if (cir[i].posx > worldSizex - cir[i].rad)
            cir[i].posx = worldSizex - cir[i].rad;
        else if (cir[i].posx < cir[i].rad)
            cir[i].posx = cir[i].rad;

        if (cir[i].posy > worldSizey - cir[i].rad)
            cir[i].posy = worldSizey - cir[i].rad;
        else if (cir[i].posy < cir[i].rad)
            cir[i].posy = cir[i].rad;
    }*/
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
    //GPU apply force maybe but prob not
    int blockSize = 256;
    int numBlocks = (numEle + blockSize - 1) / blockSize;
    applyForAll<<<numBlocks, blockSize>>>(numEle, cir, fx, fy);

    cudaDeviceSynchronize();
    /*for (unsigned i = 0; i < numEle; ++i)
    {
        cir[i].applyForce(fx, fy);
    }*/
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

/*void
PhyCuda::tempColor()
{
    for (int i = 0; i < bodies.size(); ++i)
    {
        bodies[i].red = 255;
        bodies[i].blue = 255;
        bodies[i].green = 255;
    }
}*/

//Number of cells must be divisible by 2 * threadCount for this to work
void
PhyCuda::splitCells()
{
    //Split into 4 passes
    

        
        //loop through grid and add ids
        /*unsigned* t_ids = new unsigned[numEle + (DIV * DIV)];
        unsigned* t_idLoc = new unsigned[DIV * DIV];
        unsigned index = 0;
        //unsigned sum = 0;
        for (unsigned j = 0; j < DIV; ++j)
        {
            
            for (unsigned i = 0; i < DIV; ++i)
            {
                ids[index] = grid->m_cells[i + j * DIV].m_ids.size();

                idLoc[i + j * DIV] = index;
                //sum += index;
                unsigned kMax = ids[index];

                ++index;
                for (unsigned k = 0; k < kMax; ++k)
                {
                    ids[index] = grid->m_cells[i + j * DIV].m_ids[k];
                    //std::cout << "kag;hjsdfgfhjkl;dpsa: " << ids[index] << std::endl;
                    ++index;
                }
            }
        }

        cudaMemcpy(ids, t_ids, numEle + (DIV * DIV), cudaMemcpyHostToDevice);
        cudaMemcpy(idLoc, t_idLoc, DIV * DIV, cudaMemcpyHostToDevice);
        
        delete[] t_ids;
        delete[] t_idLoc;*/
        //Construct array of pointers to m_cells.m_ids essentially
        /*unsigned **ids = new unsigned*[cellPerPass];
        for (unsigned j = 0; j < DIV; ++j)
        {
            for (unsigned i = 0; i < DIV; ++i)
            {
                int ind = i + j * DIV + DIV + 3 + (2 * j);
                ids[i + j * DIV] = &grid->m_cells[ind].m_ids[0];

                /*ids[indexIds] = new int[grid->m_cells[ind].m_ids.size()];
                for (unsigned k = 0; k < grid->m_cells[ind].m_ids.size(); ++k)
                {
                    ids[indexIds][k] = grid->m_cells[ind].m_ids[k];
                }*-/
            }
        }*/
        //cudaMallocManaged(&ids
        //cudaMallocManaged(&grid, sizeof(Grid));
        
        //cir = &bodies[0];


        int cellPerPass = DIV * DIV;
        int blockSize = 256;
        int numBlocks = (cellPerPass + blockSize - 1) / blockSize;

        //solveCell<<<numBlocks, blockSize>>>(ids, idLoc, cir, DIV);
        solveCell<<<numBlocks, blockSize>>>(cir, grid->cudaCells, DIV);
        cudaDeviceSynchronize();

        /*int maxInd = 0;
        for (int i = 0; i < bodies.size() + (DIV * DIV); ++i)
        {
            if (ids[i] > maxInd)
                maxInd = ids[i];
        }
        
        std::cout << "maxf: " << maxInd << std::endl;*/

        
    //}
}

__global__
void
//solveCell(unsigned* ids, unsigned* idLoc, CudaCircle* bodies, CudaCell* cells, const int DIV)
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

    
    /*
    int nonShiftInd = threadIdx.x + blockIdx.x * blockDim.x;
    int cellSize;
    int idInd;
    if (nonShiftInd < DIV * DIV)
    {
        idInd = idLoc[nonShiftInd];
        cellSize = ids[idInd];
    }
    else
        cellSize = 0;
    //idInd where ids at the proper cell begins
    //technically its where its size is
    
    int start = idInd + 1;
    int end = idInd + cellSize + 1;
    /*max[nonShiftInd] = cellSize;
    if (nonShiftInd == 4426)
    {
        if (cellSize != 0)
        printf("start: %d,    end: %d\n", start, end);
    }*-/

    if (cellSize != 0)
    {
        for (int i = start; i < end; ++i)
        {
            unsigned id = ids[i];
            
            //grid->m_cells[ind - 1 - (DIV + 2)]
            if (nonShiftInd / DIV != 0 && nonShiftInd % DIV != 0)
                checkEleCol(id, ids, idLoc[nonShiftInd - DIV - 1], bodies);
                
            if (nonShiftInd / DIV != 0)
                checkEleCol(id, ids, idLoc[nonShiftInd - DIV], bodies);

            if (nonShiftInd / DIV != 0 && (nonShiftInd + 1) % DIV != 0) 
                checkEleCol(id, ids, idLoc[nonShiftInd - DIV + 1], bodies);

            if (nonShiftInd % DIV != 0)
                checkEleCol(id, ids, idLoc[nonShiftInd - 1], bodies);

            checkEleCol(id, ids, idLoc[nonShiftInd], bodies);

            if ((nonShiftInd + 1) % DIV != 0)
                checkEleCol(id, ids, idLoc[nonShiftInd + 1], bodies);

            if (nonShiftInd / DIV != DIV - 1 && nonShiftInd % DIV != 0)
                checkEleCol(id, ids, idLoc[nonShiftInd + DIV - 1], bodies);

            if (nonShiftInd / DIV != DIV - 1)
            checkEleCol(id, ids, idLoc[nonShiftInd + DIV], bodies);

            if (nonShiftInd / DIV != DIV - 1 && (nonShiftInd + 1) % DIV != 0)
                checkEleCol(id, ids, idLoc[nonShiftInd + DIV + 1], bodies);
        }
    }*/
}

/*__global__
void
solveCell(Grid* grid, Circle* bodies, const int DIV)
{
    int ind = threadIdx.x + blockIdx.x * DIV + DIV + 3 + (2 * blockIdx.x);
    if (grid->m_cells[ind].m_ids.size() != 0)
    {
        for (unsigned i = 0; i < grid->m_cells[ind].m_ids.size(); ++i)
        {
            unsigned id = grid->m_cells[ind].m_ids[i];
            checkEleCol(id, grid->m_cells[ind - 1 - (DIV + 2)], bodies);
            checkEleCol(id, grid->m_cells[ind     - (DIV + 2)], bodies);
            checkEleCol(id, grid->m_cells[ind + 1 - (DIV + 2)], bodies);
            checkEleCol(id, grid->m_cells[ind             - 1], bodies);
            checkEleCol(id, grid->m_cells[ind                ], bodies);
            checkEleCol(id, grid->m_cells[ind             + 1], bodies);
            checkEleCol(id, grid->m_cells[ind - 1 + (DIV + 2)], bodies);
            checkEleCol(id, grid->m_cells[ind     + (DIV + 2)], bodies);
            checkEleCol(id, grid->m_cells[ind + 1 + (DIV + 2)], bodies);
        }
    }
}*/

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

/*__device__
void
checkEleCol(unsigned id, unsigned* ids, unsigned start, CudaCircle* bodies)
{
    int cellSize = ids[start];
    int begin = start + 1;
    int end = start + cellSize + 1;
    
        

    for (int i = begin; i < end; ++i)
    {
        solveCollision(id, ids[i], bodies);
    }
    //printf("id: %d\n", id);
}*/

__device__
void
solveCollision(unsigned i, unsigned j, CudaCircle* bodies)
{
    float epsilon = .0001f;

    /*if (i == 299)
    {
        bodies[j].red = 0;
    }*/
    if (i == j)
        return;

    
    //printf("i: %d     bodyx: %f       bodyy: %f\n", i, bodies[i].posx, bodies[i].posy);
    
    float colAxisx = bodies[i].posx - bodies[j].posx;
    float colAxisy = bodies[i].posy - bodies[j].posy;
    float distSq = colAxisx * colAxisx + colAxisy * colAxisy;
    //for poly - get line it crossed 
    // push the shape along the normal of that line
    float iRad = bodies[i].rad;
    float jRad = bodies[j].rad;
    //float jRad = bodies[i].getRad(bodies[j].pos);
    //float iRad = bodies[j].getRad(bodies[i].pos);
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

CudaCircle* 
PhyCuda::createCircle(float posx, float posy, float mass, float rad, bool pinned)
{
    //bodies.emplace_back( posx, posy, mass, rad, pinned );
    //grid->addSingle(posx, posy, bodies.size() - 1);

    CudaCircle circ (posx, posy, mass, rad, pinned);
    cudaMemcpy(&(cir[numEle]), &circ, sizeof(CudaCircle), cudaMemcpyHostToDevice);
    ++numEle;

    //grid->update(numEle);
    //grid->addSingle(posx, posy, numEle - 1);

    return {};//&cir[numEle - 1];
}

void
PhyCuda::insertToGrid(float posx, float posy, unsigned id)
{
    grid->addSingle(posx, posy, id);
}

/*std::vector<std::array<int, 4>>
PhyCuda::getGrid()
{
    return grid->getCells();
}
*/