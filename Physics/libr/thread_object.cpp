#include "thread_object.hpp"

ThreadObject::ThreadObject(unsigned id, ThreadQueue& tasks) : m_id(id), m_tasks(tasks)
{
    m_thread = std::thread (&ThreadObject::threadExecute, this);
}

void
ThreadObject::threadExecute()
{
    while (!stop)
    {
        std::function<void()> f;
        //Busy wait on thread cause task shouldn't take long to come in
        while (m_tasks.dequeue(f))
        {
            //Execute what is sent through
            f();
            m_tasks.taskDone();
        }
    }
}

void
ThreadObject::join()
{
    stop = true;
    m_thread.join();
}