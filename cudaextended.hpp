#pragma once

#include <iostream>
#include <SFML/Graphics.hpp>
#include "renderC.hpp"
#include "Physics/PhyCuda.cuh"

const int WIDTH = 1800;
const int HEIGHT = 900;

void 
fontThing(sf::RenderWindow &window, float dt, sf::Text& text, int count);

void
zoomViewAt(sf::Vector2i pixel, sf::RenderWindow& window, float zoom, sf::View& view);

/*void
createSoft(PhyWorld* phy, float x, float y);
*/
/*void
createSoft(PhyCuda* phy, float x, float y);*/


// Starts the simulation with no spatial partitioning and with a serial physics solver
int
Cuda (int argc, char **argv)
{
    //Create threadpool
    ThreadPool tp (10);

    //PhyWorld physics = PhyWorld(WIDTH, HEIGHT);
    //PhyPSS physics (WIDTH, HEIGHT, tp);
    PhyCuda physics (WIDTH * 3, HEIGHT * 3);


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
    bool jet = false;
    int jetC = 1;
    
    int counter1 = 0;

    
    //Render rd (physics, tp);
    RenderC rd (physics, tp);


    //init font
    sf::Text fps;
    sf::Font font;
    if (!font.loadFromFile("C:\\Windows\\Fonts\\arial.ttf"))
    {}
    fps.setFont(font);
    fps.setCharacterSize(16);
    fps.setFillColor(sf::Color::Green);
    fps.setStyle(sf::Text::Bold);
    
    sf::View mainView = window.getDefaultView();

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
                sf::Vector2i pixelPos = sf::Mouse::getPosition(window);
                //convert mousePos
                sf::Vector2f mousePos = window.mapPixelToCoords(pixelPos);

                ++counter1;
                if (!spawnMode)
                    //createSoft(&physics, mousePos.x, mousePos.y);
                    //physics.createCircle(mousePos.x, mousePos.y, 1, 4);
                    jet = !jet;
                else
                {
                    for (int i = 0; i < 20; ++i)
                    {
                        for (int j = 0; j < 20; ++j)
                        {
                            physics.createCircle(static_cast<float>((i * 10) + mousePos.x), 
                                                 static_cast<float>((j * 10) + mousePos.y), 1, 4);
                        }
                    }
                }
            }
            if (sf::Event::MouseWheelScrolled == evnt.type)
            {
                if (evnt.mouseWheelScroll.delta > 0)
                    zoomViewAt({ evnt.mouseWheelScroll.x, evnt.mouseWheelScroll.y }, window, (1.f / 1.25), mainView);
                else
                    zoomViewAt({ evnt.mouseWheelScroll.x, evnt.mouseWheelScroll.y }, window, 1.25, mainView);
                //mainView.zoom(zoom);
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
        
        if (jet)
        {
            if (jetC % 4 == 0)
            {
                jetC = 1;
                for (int i = 0; i < 10; ++i)
                {
                    physics.createCircle(10, 10 + (i * 10), 1, 4, true);
                }
            }
            ++jetC;
        }

        if (isGravity)
        {
            physics.gravity = 250.f;
        }
        else
        {
            physics.gravity = 0.f;
        }
        physics.update(dt.asSeconds());

        window.clear();

        //draw
        window.setView(mainView);
        rd.draw(window);

        window.setView(window.getDefaultView());
        fontThing(window, dt.asSeconds(), fps, physics.numEle);
        window.setView(mainView);
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
fontThing(sf::RenderWindow &window, float dt, sf::Text& text, int count) 
{
    std::string poop = std::to_string(count) + "    " + std::to_string(1.f/dt);
    text.setString(poop);
    window.draw(text);
}

void
zoomViewAt(sf::Vector2i pixel, sf::RenderWindow& window, float zoom, sf::View& view)
{
	const sf::Vector2f beforeCoord{ window.mapPixelToCoords(pixel) };
	//sf::View view{ window.getView() };
	view.zoom(zoom);
	window.setView(view);
	const sf::Vector2f afterCoord{ window.mapPixelToCoords(pixel) };
	const sf::Vector2f offsetCoords{ beforeCoord - afterCoord };
	view.move(offsetCoords);
	window.setView(view);
}

