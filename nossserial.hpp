#pragma once

#include <iostream>
#include <SFML/Graphics.hpp>
#include "render.hpp"
#include "Physics/PhyWorld.hpp"
#include "Physics/PhySSS.hpp"
#include "Physics/PhyPSS.hpp"

const int WIDTH = 1920;
const int HEIGHT = 1080;

void 
fontThing(sf::RenderWindow &window, float dt, sf::Text& text);


// Starts the simulation with no spatial partitioning and with a serial physics solver
int
noSsSerial (int argc, char **argv)
{
    //Create threadpool
    ThreadPool tp (10);

    //PhyWorld physics = PhyWorld(WIDTH, HEIGHT);
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

    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "ah", sf::Style::Fullscreen);
    //window.setActive(false);

    
    //window.setFramerateLimit(120);

    
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
    


    /*sf::CircleShape back(300.f);
    back.setPointCount(500);
    back.setFillColor(sf::Color::Black);
    back.setOrigin(300.f, 300.f);
    back.setPosition(640.f, 360.f);
    */
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
                //physics.createCircle(Vec2f(mousePos.x, mousePos.y), 1, 5);
                for (int i = 0; i < 20; ++i)
                {
                    for (int j = 0; j < 20; ++j)
                    {
                        Circle pm = *physics.createCircle(Vec2f(((i * 450) / 20) + 450 + counter1, ((j * 220) / 20) + 200 + counter1), 1, 5);
                    }
                }
            }
        }
        
        //window.clear(sf::Color(128, 128, 128));
        //window.draw(back);

        //apply gravity
        

        /*sf::Vector2i mousePos = sf::Mouse::getPosition(window);
        std::vector<unsigned> range = physics.tree->query(mousePos.x, mousePos.y, 50);
        for (int i = 0; i < range.size(); ++i)
        {
            physics.bodies[range[i]].red = 0;
        }*/

        /*sf::RectangleShape tester (sf::Vector2f(320, 180));
        tester.setPosition(1, 1);
        tester.setFillColor(sf::Color::Transparent);
        tester.setOutlineColor(sf::Color::White);
        tester.setOutlineThickness(2);
        window.draw(tester);*/

        if (physics.bodies.size() > 300)
            physics.bodies[299].green = 0;

        //Draw quadTree
        std::vector<sf::RectangleShape> rectangles;
        std::vector<std::array<int, 4>> grid = physics.getGrid();


        /*for (unsigned i = 0; i < grid.size(); ++i)
        {
            rectangles.emplace_back(sf::Vector2f(grid[i][2], grid[i][3]));
            rectangles.back().setPosition(grid[i][0], grid[i][1]);
            rectangles.back().setFillColor(sf::Color::Transparent);
            rectangles.back().setOutlineColor(colors[i]);
            rectangles.back().setOutlineThickness(1);
            window.draw(rectangles.back());
        }*/


        for (unsigned i = 0; i < physics.bodies.size(); ++i)
        {
            physics.bodies[i].applyAcc(Vec2f(0, 2000));
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

void 
fontThing(sf::RenderWindow &window, float dt, sf::Text& text) 
{
    std::string poop = std::to_string(1.f/dt);
    text.setString(poop);
    window.draw(text);
}

