#pragma once
#include <vector>
#include "thread_object.hpp"

class ThreadPool
{
    public:

    ThreadPool(unsigned threadCount) : m_threadCount(threadCount)
    {
        m_threads.reserve(threadCount);
        for (unsigned i = 0; i < threadCount; ++i)
        {
            m_threads.emplace_back(i, m_tasks);
        }
    }

    template <typename callable>
    void 
    execute(callable&& fullF, unsigned count)
    {
        for (unsigned i = 0; i < m_threadCount; ++i)
        {
            unsigned begin = (i * count) / m_threadCount;
            unsigned end = ((i + 1) * count) / m_threadCount;
            addTask([begin, end, &fullF] { fullF(begin, end); });
        }
    }
    
    void
    wait()
    {
        m_tasks.waitComplete();
    }

    void
    stop()
    {
        for (auto& t : m_threads)
            t.join();
    }

    private:

    template <typename callable>
    void 
    addTask(callable&& f)
    {
        m_tasks.enqueue(std::forward<callable>(f));
    }

    
    unsigned m_threadCount;
    ThreadQueue m_tasks;
    std::vector<ThreadObject> m_threads;
};