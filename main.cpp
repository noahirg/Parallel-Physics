#include <iostream>
#include <vector>
#include <SFML/Graphics.hpp>
#include "nossserial.hpp"
#include <numeric>
#include <thread>

int
main (int argc, char **argv)
{
    //For now start nossserial but eventually take input to start it in different ways
    noSsSerial(argc, argv);
    //std::cout << std::thread::hardware_concurrency() << std::endl;

    return 0;
}