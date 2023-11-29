#include <iostream>
#include <vector>
#include <SFML/Graphics.hpp>
#include "nossserial.hpp"
//#include "SSserial.hpp"
//#include "pool.hpp"
//#include "cuda.hpp"
#include <numeric>
#include <thread>

int
main (int argc, char **argv)
{
    //For now start nossserial but eventually take input to start it in different ways
    noSsSerial(argc, argv);
    //Cuda(argc, argv);
    //SsSerial(argc, argv);
    //Pool(argc, argv);
    //std::cout << std::thread::hardware_concurrency() << std::endl;

    return 0;
}