#pragma once

#include <iostream>
#include <SFML/Graphics.hpp>
#include "render.hpp"
#include "Physics/PhyWorld.hpp"

void
AddCircle(Circle& cir, sf::VertexArray& buf);
void
AddCircleBuf(Circle& cir, sf::VertexBuffer& buf);

void
updateCir(sf::VertexArray& buf, std::vector<Circle> const & bodies);
void
updateCirBuf(sf::VertexBuffer& buf, std::vector<Circle> const & bodies);

void
redneringThread(sf::Window* window);

void 
fontThing(sf::RenderWindow &window, float dt, sf::Text& text);

void
initFont(sf::RenderWindow &window, sf::Text& text);

// Starts the simulation with no spatial partitioning and with a serial physics solver
int
noSsSerial (int argc, char **argv)
{

    PhyWorld physics = PhyWorld();

    sf::VertexArray buf (sf::Triangles, 0);
    //sf::VertexBuffer vBuf (sf::Triangles);


    sf::RenderWindow window(sf::VideoMode(1280,720), "ah", sf::Style::Close);
    //window.setActive(false);

    //sf::Thread thread(&renderingThread, &window);
    //thread.launch();
    //window.setFramerateLimit(120);

    ThreadPool tp (10);

    Render rd (physics, tp);

    int counter1 = 0;
    //init font
    sf::Text fps;
    sf::Font font;
    if (!font.loadFromFile("C:\\Windows\\Fonts\\arial.ttf"))
    {}
    fps.setFont(font);
    fps.setCharacterSize(24);
    fps.setFillColor(sf::Color::Green);
    fps.setStyle(sf::Text::Bold);


    sf::CircleShape back(300.f);
    back.setPointCount(500);
    back.setFillColor(sf::Color::Black);
    back.setOrigin(300.f, 300.f);
    back.setPosition(640.f, 360.f);

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
                for (int i = 0; i < 20; ++i)
                {
                    for (int j = 0; j < 20; ++j)
                    {
                        Circle pm = *physics.createCircle(Vec2f(((i * 450) / 20) + 450 + counter1, ((j * 220) / 20) + 200 + counter1), 1, 5);
                        /*cirs.emplace_back( pm.rad );
                        cirs.back().setPointCount(40);
                        cirs.back().setFillColor(sf::Color::White);
                        cirs.back().setOrigin(pm.rad, pm.rad);
                        cirs.back().setPosition(pm.pos.x, pm.pos.y);*/
                        //AddCircle(pm, buf);
                    }
                }
                //Circle pm = *physics.createCircle(Vec2f(mousePos.x, mousePos.y), 1, 10);
                //AddCircle(pm, buf);
            }
        }
        window.clear();
        
        //window.clear(sf::Color(128, 128, 128));
        //window.draw(back);

        //apply gravity
        for (unsigned i = 0; i < physics.bodies.size(); ++i)
        {
            physics.bodies[i].applyAcc(Vec2f(0, 2000));
        }
        physics.update(dt.asSeconds());
        

        /*if (counter1 % 100 == 0)
        {
            std::cout << counter1 << std::endl;
        }*/

        rd.draw(window);

        //window.draw(buf);
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

void
initFont(sf::RenderWindow &window, sf::Text& text)
{
    
}

void
AddCircle(Circle& cir, sf::VertexArray& buf)
{
    int trisPerCir = 12;
    float radius = cir.rad;
    //std::cout << cir.pos.x << std::endl;
    //buf.resize(buf.getVertexCount() + (trisPerCir * 3));
    //buf.append(sf::Vertex (sf::Vector2f(cir.pos.x, cir.pos.y), sf::Color::White));
    for (int j = 0; j < trisPerCir; ++j)
    {
        //Center point of circle / triangle
        buf.append( sf::Vertex(sf::Vector2f(cir.pos.x, cir.pos.y), sf::Color::White));

        //First outer point of triangle
        buf.append( sf::Vertex(sf::Vector2f(cir.pos.x + cir.rad * cos(((j) * 2 * pi) / (trisPerCir)),
                                                        cir.pos.y + cir.rad * sin(((j) * 2 * pi) / (trisPerCir))), sf::Color::White));
        

        //Second outer point of triangle
        buf.append( sf::Vertex(sf::Vector2f(cir.pos.x + cir.rad * cos(((j + 1) * 2 * pi) / (trisPerCir)),
                                                        cir.pos.y + cir.rad * sin(((j + 1) * 2 * pi) / (trisPerCir))), sf::Color::White));
    }
    //std::cout << buf.getVertexCount() << std::endl;

    //starsBuf.update(&(circles[0]));
}

/*void
AddCircleBuf(Circle& cir, sf::VertexBuffer& buf)
{
    int trisPerCir = 12;
    float radius = cir.rad;
    //std::cout << cir.pos.x << std::endl;
    //buf.resize(buf.getVertexCount() + (trisPerCir * 3));
    //buf.append(sf::Vertex (sf::Vector2f(cir.pos.x, cir.pos.y), sf::Color::White));
    for (int j = 0; j < trisPerCir; ++j)
    {
        //Center point of circle / triangle
        buf.append( sf::Vertex(sf::Vector2f(cir.pos.x, cir.pos.y), sf::Color::White));

        //First outer point of triangle
        buf.append( sf::Vertex(sf::Vector2f(cir.pos.x + cir.rad * cos(((j) * 2 * pi) / (trisPerCir)),
                                                        cir.pos.y + cir.rad * sin(((j) * 2 * pi) / (trisPerCir))), sf::Color::White));
        

        //Second outer point of triangle
        buf.append( sf::Vertex(sf::Vector2f(cir.pos.x + cir.rad * cos(((j + 1) * 2 * pi) / (trisPerCir)),
                                                        cir.pos.y + cir.rad * sin(((j + 1) * 2 * pi) / (trisPerCir))), sf::Color::White));
    }
    //std::cout << buf.getVertexCount() << std::endl;

    //starsBuf.update(&(circles[0]));
}*/

void
updateCir(sf::VertexArray& buf, std::vector<Circle> const & bodies)
{
    int trisPerCir = 12;
    
    int shapes = buf.getVertexCount() / (trisPerCir * 3);

    for (int i = 0; i < shapes; ++i)
    {
        float radius = bodies[i].rad;
        //std::cout << "testing" << std::endl;
        for (int j = 0; j < trisPerCir; ++j)
        {
            //Center point of circle / triangle
            buf[(i * trisPerCir * 3) + (j * 3)].position = sf::Vector2f(bodies[i].pos.x, bodies[i].pos.y);
            //buf[(i * trisPerCir * 3) + (j * 3)].color = m_stars[i].color;

            //First outer point of triangle
            buf[(i * trisPerCir * 3) + (j * 3) + 1].position = sf::Vector2f(bodies[i].pos.x + radius * cos(((j) * 2 * pi) / (trisPerCir)),
                                                            bodies[i].pos.y + radius * sin(((j) * 2 * pi) / (trisPerCir)));
            //buf[(i * trisPerCir * 3) + (j * 3) + 1].color = m_stars[i].color;

            //Second outer point of triangle
            buf[(i * trisPerCir * 3) + (j * 3) + 2].position = sf::Vector2f(bodies[i].pos.x + radius * cos(((j + 1) * 2 * pi) / (trisPerCir)),
                                                            bodies[i].pos.y + radius * sin(((j + 1) * 2 * pi) / (trisPerCir)));
            //buf[(i * trisPerCir * 3) + (j * 3) + 2].color = m_stars[i].color;
        }
    }
}

void
updateCirBuf(sf::VertexBuffer& buf, std::vector<Circle> const & bodies)
{
    int trisPerCir = 12;
    
    int shapes = buf.getVertexCount() / (trisPerCir * 3);

    std::vector<sf::Vertex> newCir (trisPerCir * 3 * bodies.size());

    for (int i = 0; i < shapes; ++i)
    {
        float radius = bodies[i].rad;
        //std::cout << "testing" << std::endl;
        for (int j = 0; j < trisPerCir; ++j)
        {
            //Center point of circle / triangle
            newCir[(i * trisPerCir * 3) + (j * 3)].position = sf::Vector2f(bodies[i].pos.x, bodies[i].pos.y);
            //buf[(i * trisPerCir * 3) + (j * 3)].color = m_stars[i].color;

            //First outer point of triangle
            newCir[(i * trisPerCir * 3) + (j * 3) + 1].position = sf::Vector2f(bodies[i].pos.x + radius * cos(((j) * 2 * pi) / (trisPerCir)),
                                                            bodies[i].pos.y + radius * sin(((j) * 2 * pi) / (trisPerCir)));
            //buf[(i * trisPerCir * 3) + (j * 3) + 1].color = m_stars[i].color;

            //Second outer point of triangle
            newCir[(i * trisPerCir * 3) + (j * 3) + 2].position = sf::Vector2f(bodies[i].pos.x + radius * cos(((j + 1) * 2 * pi) / (trisPerCir)),
                                                            bodies[i].pos.y + radius * sin(((j + 1) * 2 * pi) / (trisPerCir)));
            //buf[(i * trisPerCir * 3) + (j * 3) + 2].color = m_stars[i].color;
        }
    }
    buf.update(&(newCir[0]), newCir.size(), 0);
}

void
renderingThread(sf::Window* window)
{
    window->setActive(true);

    while (window->isOpen())
    {

        window->display();
    }
}

void
initTexture()
{
    sf::Texture texture;
    texture.loadFromFile("circle.png");
    texture.generateMipmap();
    texture.setSmooth(true);
}