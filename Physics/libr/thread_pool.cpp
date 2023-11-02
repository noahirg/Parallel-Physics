#include "thread_pool.hpp"

ThreadPool::ThreadPool(unsigned threadCount) : m_threadCount(threadCount)
{
    m_threads.reserve(threadCount);
    for (unsigned i = 0; i < threadCount; ++i)
    {
        m_threads.emplace_back(i, m_tasks);
    }
}

void
ThreadPool::execute(std::function<void(unsigned, unsigned)>&& fullF, unsigned count)
{
    for (unsigned i = 0; i < m_threadCount; ++i)
    {
        unsigned begin = (i * count) / m_threadCount;
        unsigned end = ((i + 1) * count) / m_threadCount;
        addTask([begin, end, &fullF] { fullF(begin, end); });
    }
}

void
ThreadPool::addTask(std::function<void()>&& f)
{
    m_tasks.enqueue(f);
}

void
ThreadPool::stop()
{
    std::function<void()> empty;
    m_tasks.enqueue(empty);
}