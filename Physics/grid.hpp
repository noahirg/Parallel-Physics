#pragma once
#include <vector>
#include <algorithm>
#include "Circle.hpp"
#include "PhyWorld.hpp"

const int DIV = 100;

class Cell
{
    public:
    Cell(int x, int y, int width, int height, PhyWorld* phy)
        : m_x(x), m_y(y), m_width(width), m_height(height), phy(phy)
    {}

    void
    add(float x, float y, unsigned id)
    {
        m_ids.push_back(id);
    }

    void
    update()
    {
        //No updating if no elements
        if (m_ids.size() == 0)
            return;

        std::vector<int> tempRem;
        //Make sure elements still in this cell
        for (unsigned i = 0; i < m_ids.size(); ++i)
        {
            //Need to account for deleted circles
            //Can add id to circle class and check if it exists still


            //If not in cell remove
            if (!inCell(phy->bodies[m_ids[i]].pos.x, phy->bodies[m_ids[i]].pos.y))
            {
                tempRem.push_back(m_ids[i]);
            }
        }
        //Remove points not in cell
        for (unsigned i = 0; i < tempRem.size(); ++i)
        {
            m_ids.erase(std::remove(m_ids.begin(), m_ids.end(), tempRem[i]), m_ids.end());
            //Add them back into Grid
            
            phy->insertToGrid(Vec2f(phy->bodies[tempRem[i]].pos.x, phy->bodies[tempRem[i]].pos.y), tempRem[i]);
        }
    }

    bool
    intersects(float x, float y, float rad)
    {
        //find if circle intersects rectangle of Quad
        float Xn = std::max(static_cast<float>(m_x), std::min(x, static_cast<float>(m_x + m_width)));
        float Yn = std::max(static_cast<float>(m_y), std::min(y, static_cast<float>(m_y + m_height)));
        
        float Dx = Xn - x;
        float Dy = Yn - y;
        return (Dx * Dx + Dy * Dy) <= rad * rad;
    }

    bool
    rectIntersects(float x, float y, float w, float h)
    {
        return x < static_cast<float>(m_x + m_width) && (x + w) > static_cast<float>(m_x) &&
            y < static_cast<float>(m_y + m_height) && (y + h) > static_cast<float>(m_y);
    }

    bool
    inCell(float x, float y)
    {
        return (x >= m_x && x < (m_x + m_width) && y >= m_y && y < (m_y + m_height));
    }

    std::vector<unsigned>
    query(float x, float y, float rad)
    {
        //Range isn't in cell
        if (!intersects(x, y, rad))
            return {};

        //Range in cell
        return m_ids;
    }

    std::array<int, 4>
    getCell()
    {   
        return {m_x, m_y, m_width, m_height};
    }

    
    int m_x;
    int m_y;
    int m_width;
    int m_height;
    PhyWorld* phy;
    std::vector<unsigned> m_ids {};
};

class Grid
{
    public:
    Grid()
    {}

    Grid(int width, int height, PhyWorld* phy) : m_width(width), m_height(height), cellW(width / DIV),  cellH(height / DIV), phy(phy)
    {
        //Divide world into cells
        //Make sure width and height divisible by DIV
        for (int i = -1; i < DIV + 1; ++i)
        {
            for (int j = -1; j < DIV + 1; ++j)
            {
                m_cells.emplace_back(j * cellW, i * cellH, cellW, cellH, phy);
            }
        }
    }


    void
    addSingle(float x, float y, unsigned id)
    {
        //Check to which cell it goes in
        int indx = static_cast<int>(x) / cellW;
        int indy = static_cast<int>(y) / cellH;
        
        m_cells[indx + indy * DIV + DIV + 3 + (2 * indy)].add(x, y, id);
    }

    void
    initAdd()
    {
        for (unsigned i = 0; i < phy->bodies.size(); ++i)
        {
            addSingle(phy->bodies[i].pos.x, phy->bodies[i].pos.y, i);
        }
    }

    void
    update()
    {
        for (unsigned i = 0; i < m_cells.size(); ++i)
            m_cells[i].update();
    }

    std::vector<unsigned>
    query(float x, float y, float rad)
    {
        std::vector<unsigned> ids;
        for (unsigned i = 0; i < m_cells.size(); ++i)
        {
            //Get objects in relevant cells
            std::vector<unsigned> cellCon = m_cells[i].query(x, y, rad);

            ids.insert(ids.end(), cellCon.begin(), cellCon.end());
        }

        return ids;
    }

    std::vector<std::array<int, 4>>
    getCells()
    {
        std::vector<std::array<int, 4>> cellR;

        //Construct rects of all cells
        for (unsigned i = 0; i < m_cells.size(); ++i)
        {
            cellR.push_back(m_cells[i].getCell());
        }

        return cellR;
    }

    int
    getCellCount()
    {
        return m_cells.size();
    }



    //private:

    PhyWorld* phy;
    int m_width;
    int m_height;
    int cellH;
    int cellW;
    std::vector<Cell> m_cells;
};