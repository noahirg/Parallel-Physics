#pragma once
#include <SFML/Graphics.hpp>
#include "Physics/PhyCuda.cuh"
#include "Physics/libr/thread_pool.hpp"

class RenderC
{
    public:
    PhyCuda& rPhysics;
    sf::Texture rTexture;
    sf::VertexArray rObjects;
    ThreadPool& rTp;

    explicit
    RenderC(PhyCuda& physics, ThreadPool& tp) : rPhysics(physics), rObjects{sf::Quads}, rTp(tp)
    {
        initTexture();
    }

    void
    draw(sf::RenderWindow& window)
    {
        sf::RenderStates states;
        states.texture = &rTexture;
        //update particles
        updateObjects();
        rTp.wait();
        window.draw(rObjects, states);
    }


    void
    initTexture()
    {
        rTexture.loadFromFile("circle.png");
        rTexture.generateMipmap();
        rTexture.setSmooth(true);
    }


    void
    updateObjects()
    {
        rObjects.resize(rPhysics.numEle * 4);
        const float textureSize = 1024.f;


        rTp.execute( [&] (unsigned start, unsigned end) {
        for (unsigned i = start; i < end; ++i)
        {
            CudaCircle& ob = rPhysics.bodies[i];
            float radius = ob.rad;
            unsigned idx = i << 2;
            rObjects[idx + 0].position = sf::Vector2f {ob.posx, ob.posy} + sf::Vector2f {-radius, -radius};
            rObjects[idx + 1].position = sf::Vector2f {ob.posx, ob.posy} + sf::Vector2f {radius, -radius};
            rObjects[idx + 2].position = sf::Vector2f {ob.posx, ob.posy} + sf::Vector2f {radius, radius};
            rObjects[idx + 3].position = sf::Vector2f {ob.posx, ob.posy} + sf::Vector2f {-radius, radius};

            rObjects[idx + 0].texCoords = {0.f, 0.f};
            rObjects[idx + 1].texCoords = {textureSize, 0.f};
            rObjects[idx + 2].texCoords = {textureSize, textureSize};
            rObjects[idx + 3].texCoords = {0.f, textureSize};

            //Add color component to circles
            const sf::Color color (255, 255, 255);
            rObjects[idx + 0].color = color;
            rObjects[idx + 1].color = color;
            rObjects[idx + 2].color = color;
            rObjects[idx + 3].color = color;
        }}
        , rPhysics.numEle);
    }
};