#include "PhyWorld.hpp"

PhyWorld::PhyWorld(bool check) : checkCollisions(check)
{}

void
PhyWorld::update(float dt)
{
    for (int k = 0; k < ITER; ++k)
    {
        solveCollisions();
        updatePositions(dt / static_cast<float>(ITER));
        applyConstraint();
    }
}

void
PhyWorld::solveCollisions()
{
    float epsilon = .0001f;
    for (int i = 0; i < bodies.size(); ++i)
    {
        for (int j = i + 1; j < bodies.size(); ++j)
        {
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

void
PhyWorld::applyConstraint() {
    for (int i = 0; i < bodies.size(); ++i)
    {
        if (bodies[i].pos.x > 1280 - bodies[i].rad)
            bodies[i].pos.x = 1280 - bodies[i].rad;
        else if (bodies[i].pos.x < bodies[i].rad)
            bodies[i].pos.x = bodies[i].rad;

        if (bodies[i].pos.y > 720 - bodies[i].rad)
            bodies[i].pos.y = 720 - bodies[i].rad;
        else if (bodies[i].pos.y < bodies[i].rad)
            bodies[i].pos.y = bodies[i].rad;
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

void
PhyWorld::updatePositions(float dt)
{
    for (int i = 0; i < bodies.size(); ++i)
    {
        bodies[i].update(dt);
    }
}

Circle* 
PhyWorld::createCircle(Vec2f pos, float mass, float rad, bool pinned)
{
    bodies.emplace_back( pos, mass, rad, pinned );
    return &bodies.back();
}