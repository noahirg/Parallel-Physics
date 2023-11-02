#pragma once

#include <functional>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

class ThreadQueue
{
    public:

    void
    enqueue(std::function<void()>& f)
    {
        {
            std::unique_lock guard (m_mutex);
            m_tasks.push(f);
            ++m_taskCount;
        }
        m_cond.notify_one();
    }

    bool
    dequeue(std::function<void()>& f)
    {
        std::unique_lock guard (m_mutex);
        if (m_tasks.empty())
            return false;

        //move reduces copying
        f = std::move(m_tasks.front());
        m_tasks.pop();

        return true;
    }

    void
    taskDone()
    {
        --m_taskCount;
    }

    void
    waitComplete()
    {
        while (m_taskCount > 0)
        {}
    }

    private:
    std::condition_variable m_cond;
    std::mutex m_mutex;
    std::queue<std::function<void()>> m_tasks;
    std::atomic<unsigned> m_taskCount = 0;
};