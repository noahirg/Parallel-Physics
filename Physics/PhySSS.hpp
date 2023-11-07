#pragma once
#include <vector>
#include <memory>
#include "quadtree.hpp"
#include "PhyWorld.hpp"

class PhySSS : public PhyWorld
{
    public:

    PhySSS(int sizeX, int sizeY, bool check = false) : PhyWorld(sizeX, sizeY, check)
    {
        tree = new Quadtree(static_cast<int>(worldSize.x), static_cast<int>(worldSize.y), this);
    }

    void
    update(float dt)
    {
        for (int k = 0; k < ITER; ++k)
        {
            tempColor();
            solveCollisions();
            updatePositions(dt / static_cast<float>(ITER));
            applyConstraint();
            tree->update();
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
    solveCollisions()
    {
        //std::cout << "begin solColl" << std::endl;
        float epsilon = .0001f;
        for (int i = 0; i < bodies.size(); ++i)
        {
            //Runs but query still isn't right
            std::vector<unsigned> range = tree->query(bodies[i].pos.x, bodies[i].pos.y, 3 * bodies[i].rad);
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
        insertToTree(pos, bodies.size() - 1);
        //tree->addSingle(pos.x, pos.y, bodies.size() - 1);
        return &bodies.back();
    }

    void
    insertToTree(Vec2f pos, unsigned id)
    {
        tree->addSingle(pos.x, pos.y, id);
    }

    std::vector<std::array<int, 4>>
    getQuadtree()
    {
        return tree->getQuads();
    }

    Quadtree* tree;
};