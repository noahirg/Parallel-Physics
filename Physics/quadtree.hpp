#pragma once
#include <vector>
#include <algorithm>
#include "Circle.hpp"
#include "PhyWorld.hpp"

const int MAX_NUM = 8;
const int MAX_DEPTH = 4;

class Quad
{
    public:
    Quad(int x, int y, int width, int height, unsigned depth, PhyWorld* phy)
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
            //std::cout << "?: " << m_ids.size() << std::endl;
            if (static_cast<int>(m_ids.size()) + 1 > MAX_NUM)
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
        int quadIDx = static_cast<int>(2 * (x - m_x)) > m_width;
        int quadIDy = static_cast<int>(2 * (y - m_y)) > m_height;
        int indx = quadIDx + quadIDy * 2;
        children[indx]->add(x, y, id);
    }

    bool
    split()
    {
        if (m_depth >= MAX_DEPTH)
            return false;

        //Create children
        int widT = m_width / 2;
        int heiT = m_height / 2;
        children[0] = new Quad(m_x, m_y, widT, heiT, m_depth + 1, phy);
        children[1] = new Quad(m_x + widT, m_y, widT, heiT, m_depth + 1, phy);
        children[2] = new Quad(m_x, m_y + heiT, widT, heiT, m_depth + 1, phy);
        children[3] = new Quad(m_x + widT, m_y + heiT, widT, heiT, m_depth + 1, phy);

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
        {
            //**********This part definitely isn't right yet************
            std::vector<int> tempRem;
            //Make sure elements still in this quad
            for (unsigned i = 0; i < m_ids.size(); ++i)
            {
                //If not in quad remove
                if (!inQuad(phy->bodies[m_ids[i]].pos.x, phy->bodies[m_ids[i]].pos.y))
                {
                    tempRem.push_back(m_ids[i]);
                }
            }
            //Remove points not in quad
            for (unsigned i = 0; i < tempRem.size(); ++i)
            {
                m_ids.erase(std::remove(m_ids.begin(), m_ids.end(), tempRem[i]), m_ids.end());
                //Add the quads back into tree
                phy->insertToTree(Vec2f(phy->bodies[tempRem[i]].pos.x, phy->bodies[tempRem[i]].pos.y), tempRem[i]);
            }

            return m_ids.size();
        }

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
        return childSum;
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
    inQuad(float x, float y)
    {
        return (x >= m_x && x < (m_x + m_width) && y >= m_y && y < (m_y + m_height));
    }

    std::vector<unsigned>
    query(float x, float y, float rad)
    {
        std::vector<unsigned> found;
        if (!intersects(x, y, rad))
        {
            //Range isn't in quad
            return found;
        }
        //If no children return ids
        if (children[0] == nullptr)
        {
            return m_ids;
        }
        //If there are children combine m_ids
        std::vector<unsigned> ch0 = children[0]->query(x, y, rad);
        std::vector<unsigned> ch1 = children[1]->query(x, y, rad);
        std::vector<unsigned> ch2 = children[2]->query(x, y, rad);
        std::vector<unsigned> ch3 = children[3]->query(x, y, rad);
        found.insert(found.end(), ch0.begin(), ch0.end());
        found.insert(found.end(), ch1.begin(), ch1.end());
        found.insert(found.end(), ch2.begin(), ch2.end());
        found.insert(found.end(), ch3.begin(), ch3.end());

        return found;
    }

    std::vector<std::array<int, 4>>
    getQuad()
    {
        std::vector<std::array<int, 4>> list;
        list.push_back(std::array<int, 4>{m_x, m_y, m_width, m_height});
        //No children
        if (children[0] == nullptr)
            return list;

        //Children
        std::vector<std::array<int, 4>> ch0 = children[0]->getQuad();
        std::vector<std::array<int, 4>> ch1 = children[1]->getQuad();
        std::vector<std::array<int, 4>> ch2 = children[2]->getQuad();
        std::vector<std::array<int, 4>> ch3 = children[3]->getQuad();
        list.insert(list.end(), ch0.begin(), ch0.end());
        list.insert(list.end(), ch1.begin(), ch1.end());
        list.insert(list.end(), ch2.begin(), ch2.end());
        list.insert(list.end(), ch3.begin(), ch3.end());

        return list;
    }

    
    int m_x;
    int m_y;
    int m_width;
    int m_height;
    int m_depth;
    PhyWorld* phy;
    Quad* children[4] = {nullptr, nullptr, nullptr, nullptr};
    std::vector<unsigned> m_ids {};
};

class Quadtree
{
    public:
    Quadtree()
    {}

    Quadtree(int width, int height, PhyWorld* phy) : m_width(width), m_height(height), phy(phy)
    {
        m_root = new Quad(0, 0, m_width, m_height, 0, phy);
    }

    ~Quadtree()
    {
        delete m_root;
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
        //Empty all children of root and root
        /*delete m_root->children[0];
        delete m_root->children[1];
        delete m_root->children[2];
        delete m_root->children[3];
        m_root->children[0] = nullptr;
        m_root->children[1] = nullptr;
        m_root->children[2] = nullptr;
        m_root->children[3] = nullptr;

        m_root->m_ids.clear();
        
        //Add all particles back in
        initAdd();*/
        
        //Eventually will need to account for deleted objects
        m_root->update();
        
    }

    std::vector<unsigned>
    query(float x, float y, float rad)
    {
        return m_root->query(x, y, rad);
    }

    std::vector<std::array<int, 4>>
    getQuads()
    {
        return m_root->getQuad();
    }





    private:

    PhyWorld* phy;
    int m_width;
    int m_height;
    Quad* m_root;
};