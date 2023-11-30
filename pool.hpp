#pragma once

#include <iostream>
#include <SFML/Graphics.hpp>
#include "render.hpp"
#include "Physics/PhyPSS.hpp"

const int WIDTH = 1800;
const int HEIGHT = 900;

void 
fontThing(sf::RenderWindow &window, float dt, sf::Text& text);

/*void
createSoft(PhyWorld* phy, float x, float y);
*/
/*void
createSoft(PhyCuda* phy, float x, float y);*/


// Starts the simulation with no spatial partitioning and with a serial physics solver
int
Pool (int argc, char **argv)
{
    //Create threadpool
    ThreadPool tp (10);

    //PhyWorld physics = PhyWorld(WIDTH, HEIGHT);
    //PhyPSS physics (WIDTH, HEIGHT, tp);
    PhyPSS physics (WIDTH, HEIGHT, tp);


    std::vector<sf::Color> colors {sf::Color::Black, sf::Color::White, sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Yellow,
    sf::Color::Magenta, sf::Color::Cyan, sf::Color::Black, sf::Color::White, sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Yellow,
    sf::Color::Magenta, sf::Color::Cyan, sf::Color::Black, sf::Color::White, sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Yellow,
    sf::Color::Magenta, sf::Color::Cyan, sf::Color::Black, sf::Color::White, sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Yellow,
    sf::Color::Magenta, sf::Color::Cyan, sf::Color::Black, sf::Color::White, sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Yellow,
    sf::Color::Magenta, sf::Color::Cyan, sf::Color::Black, sf::Color::White, sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Yellow,
    sf::Color::Magenta, sf::Color::Cyan, sf::Color::Black, sf::Color::White, sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Yellow,
    sf::Color::Magenta, sf::Color::Cyan, sf::Color::Black, sf::Color::White, sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Yellow,
    sf::Color::Magenta, sf::Color::Cyan, sf::Color::Black, sf::Color::White, sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Yellow,
    sf::Color::Magenta, sf::Color::Cyan, sf::Color::Black, sf::Color::White, sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Yellow,
    sf::Color::Magenta, sf::Color::Cyan, sf::Color::Black, sf::Color::White, sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Yellow,
    sf::Color::Magenta, sf::Color::Cyan, sf::Color::Black, sf::Color::White, sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Yellow,
    sf::Color::Magenta, sf::Color::Cyan, sf::Color::Black, sf::Color::White, sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Yellow,
    sf::Color::Magenta, sf::Color::Cyan, sf::Color::Black, sf::Color::White, sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Yellow,
    sf::Color::Magenta, sf::Color::Cyan, sf::Color::Black, sf::Color::White, sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Yellow,
    sf::Color::Magenta, sf::Color::Cyan, sf::Color::Black, sf::Color::White, sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Yellow,
    sf::Color::Magenta, sf::Color::Cyan, sf::Color::Black, sf::Color::White, sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Yellow};

    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "ah", sf::Style::Close);
    window.setFramerateLimit(120);

    bool isGravity = false;
    bool spawnMode = true;
    
    int counter1 = 0;

    
    Render rd (physics, tp);


    //init font
    sf::Text fps;
    sf::Font font;
    if (!font.loadFromFile("C:\\Windows\\Fonts\\arial.ttf"))
    {}
    fps.setFont(font);
    fps.setCharacterSize(24);
    fps.setFillColor(sf::Color::Green);
    fps.setStyle(sf::Text::Bold);
    


    sf::Clock clock;

    while(window.isOpen()) 
    {
        sf::Time dt = clock.restart();

        sf::Event evnt;
        while(window.pollEvent(evnt)) {
            if (sf::Event::Closed == evnt.type) 
            {
                std::cout << "Closing Program" << std::endl;
                window.close();
            }
            else if (sf::Event::MouseButtonPressed == evnt.type) {
                sf::Vector2i mousePos = sf::Mouse::getPosition(window);
                ++counter1;
                if (!spawnMode)
                    //createSoft(&physics, mousePos.x, mousePos.y);
                    physics.createCircle(Vec2f (mousePos.x, mousePos.y), 1, 4);
                else
                {
                    for (int i = 0; i < 20; ++i)
                    {
                        for (int j = 0; j < 20; ++j)
                        {
                            physics.createCircle(Vec2f (static_cast<float>(((i * 450) / 20) + 450 + counter1), 
                                                 static_cast<float>(((j * 220) / 20) + 200 + counter1)), 1, 4);
                        }
                    }
                }
            }
            else if (evnt.type == sf::Event::KeyPressed)
            {
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
                {
                    isGravity = !isGravity;
                }
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::P))
                {
                    spawnMode = !spawnMode;
                }
            }
        }
        
        

        if (isGravity)
        {
            physics.gravity = {0.f, 250.f};
        }
        else
        {
            physics.gravity = {0.f, 0.f};
        }
        physics.update(dt.asSeconds());

        window.clear();

        //draw
        rd.draw(window);

        fontThing(window, dt.asSeconds(), fps);
        window.display();
        
    }

    tp.stop();
    return 0;
}

/*void
createSoft(PhyWorld* phy, float x, float y)
{
    int dim = 10;
    std::vector<int> cirs;
    //Creat circles
    for (int i = 0; i < dim; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            //if (i == 0 && j == 0)
            //    phy->createCircle(Vec2f(x + (i * 10), y + (j * 10)), 1, 4, true);
            //else
                phy->createCircle(Vec2f(x + (i * 10), y + (j * 10)), 1, 4);
            cirs.push_back( phy->bodies.size() - 1 );
        }
    }

    //Connect circles
    for (int i = 0; i < dim; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            //Right connection
            if (i != dim - 1)
            {
                phy->createJoint(10, cirs[i + j * dim], cirs[i + 1 + j * dim]);
            }
            //Down connection
            if (j != dim - 1)
            {
                phy->createJoint(10, cirs[i + j * dim], cirs[i + (j + 1) * dim]);
            }
            //Down and right connection
            if (j != dim - 1 && i != dim - 1)
            {
                phy->createJoint(14.1421356237, cirs[i + j * dim], cirs[i + 1 + (j + 1) * dim]);
            }
        }
    }
}*/

/*void
createSoft(PhyCuda* phy, float x, float y)
{
    int dim = 10;
    std::vector<int> cirs;
    //Creat circles
    for (int i = 0; i < dim; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            //if (i == 0 && j == 0)
            //    phy->createCircle(Vec2f(x + (i * 10), y + (j * 10)), 1, 4, true);
            //else
            phy->createCircle(x + (i * 10), y + (j * 10), 1, 4);
        }
    }

    //Connect circles
    for (int i = 0; i < dim; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            //Right connection
            if (i != dim - 1)
            {
                phy->createJoint(10, cirs[i + j * dim], cirs[i + 1 + j * dim]);
            }
            //Down connection
            if (j != dim - 1)
            {
                phy->createJoint(10, cirs[i + j * dim], cirs[i + (j + 1) * dim]);
            }
            //Down and right connection
            if (j != dim - 1 && i != dim - 1)
            {
                phy->createJoint(14.1421356237, cirs[i + j * dim], cirs[i + 1 + (j + 1) * dim]);
            }
        }
    }
}*/

void 
fontThing(sf::RenderWindow &window, float dt, sf::Text& text) 
{
    std::string poop = std::to_string(1.f/dt);
    text.setString(poop);
    window.draw(text);
}

