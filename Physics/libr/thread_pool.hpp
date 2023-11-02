#pragma once
#include <vector>
#include "thread_object.hpp"

class ThreadPool
{
    public:
    ThreadPool(unsigned threadCount);
    void execute(std::function<void(unsigned, unsigned)>&& fullF, unsigned count);
    void addTask(std::function<void()>&& f);
    void stop();

    private:

    
    unsigned m_threadCount;
    ThreadQueue m_tasks;
    std::vector<ThreadObject> m_threads;
};