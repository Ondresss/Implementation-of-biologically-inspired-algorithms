#include <matplot/matplot.h>
#include "TestRun.h"


int main(void) {
    //TestRun::runTestSomaAllToOne();
    //TestRun::runTestParticleSwarmOptimization();
    //TestRun::runTestAntColonyOptimization();

    TestRun::runTestFireFly();
    matplot::show();
    std::exit(EXIT_SUCCESS);
}
