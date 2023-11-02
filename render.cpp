#include "render.hpp"

Render::Render(PhyWorld& physics, ThreadPool& tp) : rPhysics(physics), rObjects{sf::Quads}, rTp(tp)
{
    initTexture();
}

void
Render::draw(sf::RenderWindow& window)
{
    sf::RenderStates states;
    states.texture = &rTexture;
    //update particles
    updateObjects();
    rTp.wait();
    window.draw(rObjects, states);
}

void
Render::initTexture()
{
    rTexture.loadFromFile("circle.png");
    rTexture.generateMipmap();
    rTexture.setSmooth(true);
}

void
Render::updateObjects()
{
    rObjects.resize(rPhysics.bodies.size() * 4);
    const float textureSize = 1024.f;
    

    //Can thread this
    rTp.execute( [&] (unsigned start, unsigned end) {
    for (unsigned i = start; i < end; ++i)
    {
        Circle& ob = rPhysics.bodies[i];
        float radius = ob.rad;
        unsigned idx = i << 2;
        rObjects[idx + 0].position = sf::Vector2f {ob.pos.x, ob.pos.y} + sf::Vector2f {-radius, -radius};
        rObjects[idx + 1].position = sf::Vector2f {ob.pos.x, ob.pos.y} + sf::Vector2f {radius, -radius};
        rObjects[idx + 2].position = sf::Vector2f {ob.pos.x, ob.pos.y} + sf::Vector2f {radius, radius};
        rObjects[idx + 3].position = sf::Vector2f {ob.pos.x, ob.pos.y} + sf::Vector2f {-radius, radius};

        rObjects[idx + 0].texCoords = {0.f, 0.f};
        rObjects[idx + 1].texCoords = {textureSize, 0.f};
        rObjects[idx + 2].texCoords = {textureSize, textureSize};
        rObjects[idx + 3].texCoords = {0.f, textureSize};

        //Add color component to circles
        const sf::Color color = sf::Color::White;
        rObjects[idx + 0].color = color;
        rObjects[idx + 1].color = color;
        rObjects[idx + 2].color = color;
        rObjects[idx + 3].color = color;
    }}
    , rPhysics.bodies.size());
}