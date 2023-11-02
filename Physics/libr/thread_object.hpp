#pragma once
#include <thread>
#include <iostream>
#include "thread_queue.hpp"


class ThreadObject
{
    public:
    ThreadObject(unsigned id, ThreadQueue& tasks);
    void threadExecute();
    void join();

    unsigned m_id;
    std::thread m_thread;
    ThreadQueue& m_tasks;
    bool stop = false;
};