#include "thread_object.hpp"

ThreadObject::ThreadObject(unsigned id, ThreadQueue& tasks) : m_id(id), m_tasks(tasks)
{
    m_thread = std::thread (&ThreadObject::threadExecute, this);
}

void
ThreadObject::threadExecute()
{
    std::function<void()> f;
    //Busy wait on thread cause task shouldn't take long to come in
    //Checks value of dequeue - if empty function returned it evaluates
    // to false and ends thread
    while ((f = m_tasks.dequeue()))
    {
        //Execute what is sent through
        std::cout << m_id << std::endl;
        f();
    }
    m_thread.join();
}