#include <matplot/matplot.h>
#include "TestRun.h"


int main(void) {
    //TestRun::runTestSomaAllToOne();
    //TestRun::runTestParticleSwarmOptimization();
    //TestRun::runTestAntColonyOptimization();

    TestRun::runTextTLBO();
    //TestRun::runXlsTest();
    std::exit(EXIT_SUCCESS);
}
