#pragma once

#include <functional>
#include <mutex>
#include <condition_variable>
#include <queue>

class ThreadQueue
{
    public:

    void
    enqueue(std::function<void()>& f)
    {
        {
            std::unique_lock guard (m_mutex);
            m_tasks.push(f);
        }
        m_cond.notify_one();
    }

    std::function<void()>
    dequeue()
    {
        std::unique_lock guard (m_mutex);
        m_cond.wait(guard, [&]{ return m_tasks.size(); });

        std::function<void()> task = m_tasks.front();
        m_tasks.pop();

        return task;
    }

    private:
    std::condition_variable m_cond;
    std::mutex m_mutex;
    std::queue<std::function<void()>> m_tasks;
};