#pragma once
#include <vector>
#include <memory>
#include "grid.hpp"
#include "PhyWorld.hpp"

class PhySSS : public PhyWorld
{
    public:

    PhySSS(int sizeX, int sizeY, bool check = false) : PhyWorld(sizeX, sizeY, check)
    {
        grid = new Grid(static_cast<int>(worldSize.x), static_cast<int>(worldSize.y), this);
    }

    ~PhySSS()
    {
        delete grid;
    }

    void
    update(float dt)
    {
        for (int k = 0; k < ITER; ++k)
        {
            tempColor();
            //solveCollisions();
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

    void
    splitCells()
    {
        //Split cells into rows to check
        for (int i = 1; i < DIV + 1; ++i)
        {
            int start = i * (DIV + 2) + 1;
            int end = i * (DIV + 2) + 1 + DIV;
            checkCells(start, end);
        }
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

        if (i == 299)
        {
            bodies[j].red = 0;
        }

        Vec2f colAxis = bodies[i].pos - bodies[j].pos;
        float distSq = colAxis.magnSq();
        //for poly - get line it crossed 
        // push the shape along the normal of that line
        float iRad = bodies[i].rad;
        float jRad = bodies[j].rad;
        //float jRad = bodies[i].getRad(bodies[j].pos);
        //float iRad = bodies[j].getRad(bodies[i].pos);
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

    void
    solveCollisions()
    {
        //std::cout << "begin solColl" << std::endl;
        float epsilon = .0001f;
        for (int i = 0; i < bodies.size(); ++i)
        {
            //Runs but query still isn't right
            std::vector<unsigned> range = grid->query(bodies[i].pos.x, bodies[i].pos.y, 2 * bodies[i].rad);
            for (int j = 0; j < range.size(); ++j)
            {
                if (i == range[j])
                    continue;

                if (i == 299)
                {
                    bodies[range[j]].red = 0;
                }

                Vec2f colAxis = bodies[i].pos - bodies[range[j]].pos;
                float distSq = colAxis.magnSq();
                //for poly - get line it crossed 
                // push the shape along the normal of that line
                float iRad = bodies[i].rad;
                float jRad = bodies[range[j]].rad;
                //float jRad = bodies[i].getRad(bodies[j].pos);
                //float iRad = bodies[j].getRad(bodies[i].pos);
                float radD = iRad + jRad;
                if (distSq < radD * radD && distSq > epsilon)
                {
                    float dist = colAxis.magn();
                    Vec2f normal = Vec2f::normalize(colAxis);
                    float delta = radD - dist;
                    float di = (jRad / radD) * delta;;
                    float dj = (iRad / radD) * delta;;

                    if (bodies[i].pinned && bodies[range[j]].pinned)
                        {di = 0; dj = 0;}
                    else if (bodies[i].pinned)
                        dj = delta;
                    else if (bodies[range[j]].pinned)
                        di = delta;

                    
                    bodies[i].pos += di * normal;
                    bodies[range[j]].pos -= dj * normal;
                }
            }
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
};