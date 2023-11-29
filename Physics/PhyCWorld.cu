#include "PhyCWorld.cuh"

PhyCWorld::PhyCWorld(int sizeX, int sizeY, bool check) : checkCollisions(check), worldSizex(sizeX), worldSizey(sizeY)
{
    bodies = new CudaCircle[5'000'000];
}

void
PhyCWorld::update(float dt)
{
    const int ITERC = 8;
    for (int k = 0; k < ITERC; ++k)
    {
        //tempColor();
        updateJoints(dt / static_cast<float>(ITERC));
        updatePositions(dt / static_cast<float>(ITERC));
        applyConstraint();
    }
}

void
PhyCWorld::applyConstraint() {
    for (int i = 0; i < 0; ++i)
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

void
PhyCWorld::updateJoints(float dt)
{
    for (unsigned i = 0; i < joints.size(); ++i)
    {
        joints[i].update(&bodies[joints[i].cir1], &bodies[joints[i].cir2], dt);
    }
}

void
PhyCWorld::updatePositions(float dt)
{
    for (int i = 0; i < 0; ++i)
    {
        //bodies[i].update(dt);
    }
}

CudaCircle* 
PhyCWorld::createCircle(float posx, float posy, float mass, float rad, bool pinned)
{
    return {};
}

CudaJoint*
PhyCWorld::createJoint(float length, int cir1, int cir2)
{
    joints.emplace_back( length, cir1, cir2 );
    return &joints.back();
}

void
PhyCWorld::insertToGrid(float posx, float posy, unsigned id)
{
    //empty on purpose
    return;
}