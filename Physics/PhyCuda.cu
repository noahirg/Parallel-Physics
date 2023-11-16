#include "PhyCuda.cuh"


__global__
void
solveCell(unsigned* ids, unsigned* idLoc, CudaCircle* bodies, const int DIV, int* max);

__device__
void
checkEleCol(unsigned id, unsigned* ids, unsigned start, CudaCircle* bodies);

__device__
void
solveCollision(unsigned i, unsigned j, CudaCircle* bodies);



PhyCuda::PhyCuda(int sizeX, int sizeY, bool check) : PhyCWorld(sizeX, sizeY, check)
{
    grid = new CudaGrid(static_cast<int>(worldSizex), static_cast<int>(worldSizey), this);
}

PhyCuda::~PhyCuda()
{
    delete grid;
}

void
PhyCuda::update(float dt)
{
    const int ITERC = 8;
    for (int k = 0; k < ITERC; ++k)
    {
        //tempColor();
        splitCells();
        updateJoints(dt / static_cast<float>(ITERC));
        updatePositions(dt / static_cast<float>(ITERC));
        applyConstraint();
        grid->update();
    }
}

void
PhyCuda::updateJoints(float dt)
{
    for (unsigned i = 0; i < joints.size(); ++i)
    {
        joints[i].update(&bodies[joints[i].cir1], &bodies[joints[i].cir2], dt);
    }
}

void
PhyCuda::updatePositions(float dt)
{
    for (int i = 0; i < bodies.size(); ++i)
    {
        bodies[i].update(dt);
    }
}

void
PhyCuda::applyConstraint() {
    for (int i = 0; i < bodies.size(); ++i)
    {
        if (bodies[i].posx > worldSizex - bodies[i].rad)
            bodies[i].posx = worldSizex - bodies[i].rad;
        else if (bodies[i].posx < bodies[i].rad)
            bodies[i].posx = bodies[i].rad;

        if (bodies[i].posy > worldSizey - bodies[i].rad)
            bodies[i].posy = worldSizey - bodies[i].rad;
        else if (bodies[i].posy < bodies[i].rad)
            bodies[i].posy = bodies[i].rad;
        /*Vec2f position = Vec2f(640.f, 360.f);
        float radius = 300.f;
        Vec2f toObj = bodies[i].pos - position;
        float distSq = toObj.magnSq();
        if (distSq > (radius - bodies[i].rad) * (radius - bodies[i].rad)) {
            float dist = toObj.magn();
            Vec2f n = toObj / dist;
            bodies[i].pos = position + n * (radius - bodies[i].rad);
        }*/
    }
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
    int cellPerPass = DIV * DIV;// / 4;
    //For loop for each pass
    //for (unsigned i = 0; i < 4; ++i)
    //{
        //Solve each cell individually instead of in groups like pool
        //Grid* gr = grid;

        //Construct contiguous array that holds all ids
        //Beginning of id block has the size of the particular cell
        unsigned *ids = new unsigned[bodies.size() + (DIV * DIV)];
        unsigned *idLoc = new unsigned[DIV * DIV]();

        cudaMallocManaged(&ids, bodies.size() + (DIV * DIV));
        cudaMallocManaged(&idLoc, DIV * DIV);
        //loop through grid and add ids
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
        CudaCircle* cir;
        cudaMallocManaged(&cir, bodies.size() * sizeof(CudaCircle));

        cudaMemcpy(cir, &bodies[0], bodies.size() * sizeof(CudaCircle), cudaMemcpyHostToDevice);
        //cir = &bodies[0];



        int blockSize = 256;
        int numBlocks = (cellPerPass + blockSize - 1) / blockSize;
        int* max = new int[blockSize * numBlocks]();
        cudaMallocManaged(&max, sizeof(int));
        solveCell<<<numBlocks, blockSize>>>(ids, idLoc, cir, DIV, max);
        cudaDeviceSynchronize();

        cudaMemcpy(&bodies[0], cir, bodies.size() * sizeof(CudaCircle), cudaMemcpyDeviceToHost);

        /*int maxInd = 0;
        for (int i = 0; i < bodies.size() + (DIV * DIV); ++i)
        {
            if (ids[i] > maxInd)
                maxInd = ids[i];
        }
        
        std::cout << "maxf: " << maxInd << std::endl;*/

        cudaFree(cir);
        cudaFree(ids);
        cudaFree(idLoc);
        cudaFree(max);
    //}
}

__global__
void
solveCell(unsigned* ids, unsigned* idLoc, CudaCircle* bodies, const int DIV, int* max)
{
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
        //printf("nonShiftInd: %d\n", nonShiftInd);
        //printf("fuck yourself\n");
        //printf("cellsize: %d\n", cellSize);
        if (cellSize != 0)
        printf("start: %d,    end: %d\n", start, end);
    }*/

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
    }
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
}

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
    bodies.emplace_back( posx, posy, mass, rad, pinned );
    //insertToGrid(pos, bodies.size() - 1);
    grid->addSingle(posx, posy, bodies.size() - 1);
    //tree->addSingle(pos.x, pos.y, bodies.size() - 1);
    return &bodies.back();
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