#pragma once
#include <SFML/Graphics.hpp>
#include "Physics/PhyWorld.hpp"
#include "Physics/libr/thread_pool.hpp"

class Render
{
    public:
    PhyWorld& rPhysics;
    sf::Texture rTexture;
    sf::VertexArray rObjects;
    ThreadPool& rTp;

    explicit
    Render(PhyWorld& physics, ThreadPool& tp);
    void draw(sf::RenderWindow& window);
    void initTexture();
    void updateObjects();
};