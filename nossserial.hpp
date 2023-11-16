#pragma once

#include <iostream>
#include <SFML/Graphics.hpp>
//#include "render.hpp"
#include "renderC.hpp"
/*#include "Physics/PhyWorld.hpp"
#include "Physics/PhySSS.hpp"
#include "Physics/PhyPSS.hpp"*/
#include "Physics/PhyCuda.cuh"

const int WIDTH = 1800;
const int HEIGHT = 900;

void 
fontThing(sf::RenderWindow &window, float dt, sf::Text& text);

/*void
createSoft(PhyWorld* phy, float x, float y);
*/
void
createSoft(PhyCWorld* phy, float x, float y);


// Starts the simulation with no spatial partitioning and with a serial physics solver
int
noSsSerial (int argc, char **argv)
{
    //Create threadpool
    ThreadPool tp (10);

    //PhyWorld physics = PhyWorld(WIDTH, HEIGHT);
    //PhyPSS physics (WIDTH, HEIGHT, tp);
    PhyCuda physics (WIDTH, HEIGHT);


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
    //window.setActive(false);

    //window.setFramerateLimit(120);
    bool isGravity = true;
    bool spawnMode = true;
    
    int counter1 = 0;

    
    //Render rd (physics, tp);
    RenderC rd (physics, tp);


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

    /*Circle* pair1;
    Circle* pair2;*/
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
                //if (counter1 == 1)
                //{
                if (!spawnMode)
                    createSoft(&physics, mousePos.x, mousePos.y);
                else
                {
                    for (int i = 0; i < 20; ++i)
                    {
                        for (int j = 0; j < 20; ++j)
                        {
                            physics.createCircle(static_cast<float>(((i * 450) / 20) + 450 + counter1), 
                                                 static_cast<float>(((j * 220) / 20) + 200 + counter1), 1, 4);
                        }
                    }
                }

                //}
                //else
                //{
                    /*if (counter1 == 2)
                        pair1 = physics.createCircle(Vec2f(mousePos.x, mousePos.y), 1, 4);
                    else if (counter1 == 3)
                    {
                        pair2 = physics.createCircle(Vec2f(mousePos.x, mousePos.y), 1, 4);
                        physics.createJoint(10, pair1, pair2);
                    }
                    else
                        physics.createCircle(Vec2f(mousePos.x, mousePos.y), 1, 4);*/
                //}
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
        /*if (physics.bodies.size() > 1)
        {
            std::cout << physics.bodies[0].pos.x - physics.bodies[0].posOld.x << "\t\t" << physics.bodies[0].pos.y - physics.bodies[0].posOld.y;
            std::cout << "\t\t\t\t" << physics.bodies[1].pos.x - physics.bodies[1].posOld.x << "\t\t" << physics.bodies[1].pos.y - physics.bodies[1].posOld.y << std::endl;
        }*/
        /*if (physics.joints.size() > 0)
        {
            std::cout << physics.joints[0].prevForce.x << "\t\t" << physics.joints[0].prevForce.y << std::endl;
        }*/

        /*if (physics.bodies.size() > 300)
            physics.bodies[299].green = 0;*/

        //Draw quadTree
        /*std::vector<sf::RectangleShape> rectangles;
        std::vector<std::array<int, 4>> grid = physics.getGrid();*/


        /*for (unsigned i = 0; i < grid.size(); ++i)
        {
            rectangles.emplace_back(sf::Vector2f(grid[i][2], grid[i][3]));
            rectangles.back().setPosition(grid[i][0], grid[i][1]);
            rectangles.back().setFillColor(sf::Color::Transparent);
            rectangles.back().setOutlineColor(colors[i]);
            rectangles.back().setOutlineThickness(1);
            window.draw(rectangles.back());
        }*/

        //Draw links
        /*sf::VertexArray links (sf::Lines, 0);
        for (unsigned i = 0; i < physics.joints.size(); ++i)
        {
            links.append( {{physics.bodies[physics.joints[i].cir1].pos.x, physics.bodies[physics.joints[i].cir1].pos.y}} );
            links.append( {{physics.bodies[physics.joints[i].cir2].pos.x, physics.bodies[physics.joints[i].cir2].pos.y}} );
        }
        window.draw(links);*/
        //std::cout << "*******" << std::endl;

        if (physics.bodies.size() > 0)
        {
            int indx = static_cast<int>(physics.bodies[1].posx) / (WIDTH / DIV);
            int indy = static_cast<int>(physics.bodies[1].posy) / (HEIGHT / DIV);

            //std::cout << "indx: " << indx << "     indy: " << indy << std::endl;
        }

        if (!isGravity)
        {
            for (unsigned i = 0; i < physics.bodies.size(); ++i)
            {
                physics.bodies[i].applyAcc(0.f, 2000.f);
            }
        }
        physics.update(dt.asSeconds());
        //std::cout << "(((((((((" << std::endl;
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

void
createSoft(PhyCWorld* phy, float x, float y)
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
}

void 
fontThing(sf::RenderWindow &window, float dt, sf::Text& text) 
{
    std::string poop = std::to_string(1.f/dt);
    text.setString(poop);
    window.draw(text);
}

