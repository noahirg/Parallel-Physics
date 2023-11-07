#include "Body.hpp"

Body::Body() = default;

Body::Body(float x, float y, float mass, bool pinned) : pos(Vec2f(x, y)), mass(mass), pinned(pinned), posOld(Vec2f(x, y))
{}

Body::Body(Vec2f pos, float mass, bool pinned) : pos(pos), mass(mass), pinned(pinned), posOld(pos)
{}

void
Body::update(float dt)
{
    if (pinned) return;
    Vec2f velo = pos - posOld;

    posOld = pos;
    pos += velo + acc * dt * dt;

    acc = {};
}

void
Body::applyForce(Vec2f force) 
{
    if (pinned) return;
    if (mass == 0) {
        this->acc = {0, 0};
    }
    else {
        this->acc += force / mass;
    }
}

void
Body::applyForce(float fx, float fy) 
{
    if (pinned) return;
    if (mass == 0) {
        this->acc = {0, 0};
    }
    else {
        this->acc += {fx / mass, fy / mass};
    }
}

void
Body::applyAcc(Vec2f acc) 
{
    if (pinned) return;
    this->acc += acc;
}

void
Body::applyAcc(float accx, float accy) 
{
    if (pinned) return;
    this->acc += {accx / mass, accy / mass};
}

void
Body::pin()
{
    pinned = true;
}

std::tuple<short, short, short> 
Body::getColor()
{
    return {red, green, blue};
}