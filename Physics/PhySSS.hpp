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
        tree = new Quadtree(static_cast<unsigned>(worldSize.x), static_cast<unsigned>(worldSize.y), this);
    }

    void
    update(float dt)
    {
        for (int k = 0; k < ITER; ++k)
        {
            solveCollisions();
            updatePositions(dt / static_cast<float>(ITER));
            applyConstraint();
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
            std::vector<int> range = tree->query(bodies[i].pos.x, bodies[i].pos.y, 50);
            for (int j = 0; j < range.size(); ++j)
            {
                if (i == j)
                    continue;

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
                    float di = (jRad / radD) * delta;;
                    float dj = (iRad / radD) * delta;;

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
        }
    }

    Circle* 
    createCircle(Vec2f pos, float mass, float rad, bool pinned = false)
    {
        bodies.emplace_back( pos, mass, rad, pinned );
        tree->addSingle(pos.x, pos.y, bodies.size() - 1);
        return &bodies.back();
    }

    Quadtree* tree;
};