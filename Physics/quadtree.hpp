#pragma once
#include <vector>
#include "Circle.hpp"
#include "PhyWorld.hpp"

const unsigned MAX_NUM = 16;
const unsigned MAX_DEPTH = 3;

class Quad
{
    public:
    Quad(unsigned x, unsigned y, unsigned width, unsigned height, unsigned depth, PhyWorld* phy)
        : m_x(x), m_y(y), m_width(width), m_height(height), m_depth(depth), phy(phy)
    {}

    ~Quad()
    {
        delete children[0];
        delete children[1];
        delete children[2];
        delete children[3];
    }

    void
    add(float x, float y, unsigned id)
    {
        //If no children
        if (children[0] == nullptr)
        {
            if (m_ids.size() > MAX_NUM)
            {
                //split
                if (split())
                {
                    addToChild(x, y, id);
                    return;
                }
            }
            m_ids.push_back(id);
            return;
        }
        //Has children
        addToChild(x, y, id);
    }

    void
    addToChild(float x, float y, unsigned id)
    {
        int quadIDx = 2 * x > m_width;
        int quadIDy = 2 * y > m_height;
        int indx = quadIDx + quadIDy * 2;
        children[indx]->add(x, y, id);
    }

    bool
    split()
    {
        if (m_depth >= MAX_DEPTH)
            return false;

        //Create children
        children[0] = new Quad(m_x, m_y, m_width / 2, m_height / 2, m_depth + 1, phy);
        children[1] = new Quad(m_width / 2, m_y, m_width, m_height / 2, m_depth + 1, phy);
        children[2] = new Quad(m_x, m_height / 2, m_width / 2, m_height, m_depth + 1, phy);
        children[3] = new Quad(m_width / 2, m_height / 2, m_width, m_height, m_depth + 1, phy);

        //Add vector ids to children
        for (unsigned i = 0; i < m_ids.size(); ++i)
        {
            float x = phy->bodies[m_ids[i]].pos.x;
            float y = phy->bodies[m_ids[i]].pos.y;

            addToChild(x, y, m_ids[i]);
        }

        //Clear this vector
        m_ids.clear();
        return true;
    }

    unsigned
    update()
    {
        //If no children
        if (children[0] == nullptr)
            return m_ids.size();

        //If children
        unsigned childSum = 0;
        childSum += children[0]->update();
        childSum += children[1]->update();
        childSum += children[2]->update();
        childSum += children[3]->update();

        //If childSum is less than max_num then remove children
        if (childSum <= MAX_NUM)
        {
            //Combine the vectors into this main node
            m_ids.insert(m_ids.end(), children[0]->m_ids.begin(), children[0]->m_ids.end());
            m_ids.insert(m_ids.end(), children[1]->m_ids.begin(), children[1]->m_ids.end());
            m_ids.insert(m_ids.end(), children[2]->m_ids.begin(), children[2]->m_ids.end());
            m_ids.insert(m_ids.end(), children[3]->m_ids.begin(), children[3]->m_ids.end());

            //Delete children
            delete children[0];
            delete children[1];
            delete children[2];
            delete children[3];
            children[0] = nullptr;
            children[1] = nullptr;
            children[2] = nullptr;
            children[3] = nullptr;
        }
    }

    bool
    intersects(float x, float y, float rad)
    {
        //find if circle intersects rectangle
        float Xn = std::max(static_cast<float>(m_x), std::min(x, static_cast<float>(m_x + m_width)));
        float Yn = std::max(static_cast<float>(m_y), std::min(y, static_cast<float>(m_y + m_height)));
        
        float Dx = Xn - x;
        float Dy = Yn - y;
        return (Dx * Dx + Dy * Dy) <= rad * rad;
    }

    std::vector<int>
    query(float x, float y, float rad)
    {
        if (!intersects(x, y, rad))
        {
            //Range isn't in quad
            std::vector<int> empty;
            return empty;
        }

        //If no children return ids
        if (children[0] == nullptr)
        {
            return m_ids;
        }
        //If there are children combine m_ids
        std::vector<int> ch0 = children[0]->query(x, y, rad);
        std::vector<int> ch1 = children[1]->query(x, y, rad);
        std::vector<int> ch2 = children[2]->query(x, y, rad);
        std::vector<int> ch3 = children[3]->query(x, y, rad);
        m_ids.insert(m_ids.end(), ch0.begin(), ch0.end());
        m_ids.insert(m_ids.end(), ch1.begin(), ch1.end());
        m_ids.insert(m_ids.end(), ch2.begin(), ch2.end());
        m_ids.insert(m_ids.end(), ch3.begin(), ch3.end());

        return m_ids;
    }

    
    unsigned m_x;
    unsigned m_y;
    unsigned m_width;
    unsigned m_height;
    unsigned m_depth;
    PhyWorld* phy;
    Quad* children[4] = {nullptr, nullptr, nullptr, nullptr};
    std::vector<int> m_ids;
};

class Quadtree
{
    public:
    Quadtree()
    {}

    Quadtree(unsigned width, unsigned height, PhyWorld* phy) : m_width(width), m_height(height), phy(phy)
    {
        m_root = new Quad(0, 0, m_width, m_height, 0, phy);
    }

    void
    addSingle(float x, float y, unsigned id)
    {
        m_root->add(x, y, id);
    }

    void
    initAdd()
    {
        for (unsigned i = 0; i < phy->bodies.size(); ++i)
        {
            m_root->add(phy->bodies[i].pos.x, phy->bodies[i].pos.y, i);
        }
    }

    void
    update()
    {
        //Eventually will need to account for deleted objects
        m_root->update();
    }

    std::vector<int>
    query(float x, float y, float rad)
    {
        return m_root->query(x, y, rad);
    }





    private:

    PhyWorld* phy;
    unsigned m_width;
    unsigned m_height;
    Quad* m_root;
};