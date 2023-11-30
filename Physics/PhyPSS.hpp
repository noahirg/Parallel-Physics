#pragma once
#include <vector>
#include <memory>
#include "grid.hpp"
#include "PhyWorld.hpp"
#include "libr/thread_pool.hpp"

class PhyPSS : public PhyWorld
{
    public:

    PhyPSS(int sizeX, int sizeY, ThreadPool& tp, bool check = false) : PhyWorld(sizeX, sizeY, check), tp(tp)
    {
        grid = new Grid(static_cast<int>(worldSize.x), static_cast<int>(worldSize.y), this);
    }

    ~PhyPSS()
    {
        delete grid;
    }

    void
    update(float dt)
    {
        for (int k = 0; k < ITER; ++k)
        {
            tempColor();
            splitCells();
            updateJoints(dt / static_cast<float>(ITER));
            updatePositions(dt / static_cast<float>(ITER));
            applyConstraint();
            grid->update();
        }
    }

    void
    tempColor()
    {
        for (int i = 0; i < bodies.size(); ++i)
        {
            bodies[i].red = 255;
            bodies[i].blue = 255;
            bodies[i].green = 255;
        }
    }

    //Number of cells must be divisible by 2 * threadCount for this to work
    void
    splitCells()
    {
        //To prevent race, split data into two passes
        unsigned threadCount = tp.getThreadCount();
        unsigned divCount = threadCount * 2;
        unsigned divSize = (DIV * DIV) / divCount;
        unsigned sections = DIV / divCount;
        int init = DIV + 3;

        //First pass
        for (unsigned i = 0; i < threadCount; ++i)
        {
            //Split into sections so exterior border cells ignored
            tp.addTask(
                [this, i, sections, init]
                {

                    for (unsigned j = 0; j < sections; ++j)
                    {
                        unsigned start = (init * (j + 1) - j) + (DIV * i * 2 * sections + (i * 4 * sections));
                        unsigned end = start + DIV;
                        checkCells(start, end);
                    }
                });
        }
        tp.wait();
        //Second pass
        for (unsigned i = 0; i < threadCount; ++i)
        {
            tp.addTask(
                [this, i, sections, init]
                {
                    
                    for (unsigned j = 0; j < sections; ++j)
                    {
                        unsigned start = (init * (j + sections + 1) - (j + sections)) + (DIV * i * 2 * sections + (i * 4 * sections));
                        unsigned end = start + DIV;
                        checkCells(start, end);
                    }
                    
                });
        }
        tp.wait();
    }

    void
    checkCells(int start, int end)
    {
        //Iterate through cells
        for (unsigned i = start; i < end; ++i)
        {
            solveCell(i);
        }
    }

    void
    solveCell(unsigned ind)
    {
        if (grid->m_cells[ind].m_ids.size() != 0)
        {
            for (unsigned i = 0; i < grid->m_cells[ind].m_ids.size(); ++i)
            {
                unsigned id = grid->m_cells[ind].m_ids[i];
                checkEleCol(id, grid->m_cells[ind - 1 - (DIV + 2)]);
                checkEleCol(id, grid->m_cells[ind     - (DIV + 2)]);
                checkEleCol(id, grid->m_cells[ind + 1 - (DIV + 2)]);
                checkEleCol(id, grid->m_cells[ind - 1]);
                checkEleCol(id, grid->m_cells[ind]);
                checkEleCol(id, grid->m_cells[ind + 1]);
                checkEleCol(id, grid->m_cells[ind - 1 + (DIV + 2)]);
                checkEleCol(id, grid->m_cells[ind     + (DIV + 2)]);
                checkEleCol(id, grid->m_cells[ind + 1 + (DIV + 2)]);
            }
        }
    }

    void
    checkEleCol(unsigned id, Cell& c)
    {
        for (unsigned i = 0; i < c.m_ids.size(); ++i)
        {
            solveCollision(id, c.m_ids[i]);
        }
    }

    void
    solveCollision(unsigned i, unsigned j)
    {
        float epsilon = .0001f;


        Vec2f colAxis = bodies[i].pos - bodies[j].pos;
        float distSq = colAxis.magnSq();
        // push the shape along the normal of that line
        float iRad = bodies[i].rad;
        float jRad = bodies[j].rad;
        float radD = iRad + jRad;
        if (distSq < radD * radD && distSq > epsilon)
        {
            float dist = colAxis.magn();
            Vec2f normal = Vec2f::normalize(colAxis);
            float delta = radD - dist;
            float di = (jRad / radD) * delta;
            float dj = (iRad / radD) * delta;

            if (bodies[i].pinned && bodies[j].pinned)
                {di = 0; dj = 0;}
            else if (bodies[i].pinned)
                dj = delta;
            else if (bodies[j].pinned)
                di = delta;

            
            bodies[i].pos += di * normal;
            bodies[j].pos -= dj * normal;
        }
    }


    Circle* 
    createCircle(Vec2f pos, float mass, float rad, bool pinned = false)
    {
        bodies.emplace_back( pos, mass, rad, pinned );
        //insertToGrid(pos, bodies.size() - 1);
        grid->addSingle(pos.x, pos.y, bodies.size() - 1);
        //tree->addSingle(pos.x, pos.y, bodies.size() - 1);
        return &bodies.back();
    }

    void
    insertToGrid(Vec2f pos, unsigned id)
    {
        grid->addSingle(pos.x, pos.y, id);
    }


    Grid* grid;
    ThreadPool& tp;
};