#include <iostream>
#include "Physics/libr/thread_pool.hpp"

int
main()
{
    ThreadPool rTp (10);
    auto lamda = [&] (unsigned start, unsigned end) {
    for (unsigned i = start; i < end; ++i)
    {
        std::cout << "test:" << start << std::endl;
    }};
    rTp.execute( lamda, 10);
    rTp.execute( lamda, 10);
    rTp.execute( lamda, 20);

    int count = 0;
    for (unsigned i = 0; i < 1'000'000'000; ++i)
    {
        ++count;
    }
    std::cout << count << std::endl;

    rTp.stop();
    return 0;
}