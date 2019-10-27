#include <iostream>
#include<LongImageProcess.h>
#include <thread>

using namespace std;

int main()
{
    LongImageProcess process;
    std::thread t1(&LongImageProcess::LongImageProducter,process);
    std::thread t2(&LongImageProcess::ImageConsumer,process);
        t2.join();
    t1.join();
//    process.LongImageProducter();
}
