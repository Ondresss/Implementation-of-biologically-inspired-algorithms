//
// Created by andrew on 16/11/2025.
//

#pragma once
#include <memory>
#include "Solution.h"
#include <vector>
#include <xlsxwriter.h>
class XlsWriter {
public:
    XlsWriter() = default;

    XlsWriter(std::vector<std::shared_ptr<Solution>> solutions, std::vector<std::shared_ptr<Function>> testFunctions_ )
    : solutions(solutions),testFunctions(testFunctions_) {};
    void write(int noIterations, int noExperiments);

private:
    std::vector<std::shared_ptr<Solution>> solutions;
    std::vector<std::shared_ptr<Function>> testFunctions;
    [[nodiscard]] double getMean(const std::vector<double>& vals) const;
    [[nodiscard]] double getStdDev(const std::vector<double>& vals) const;
};
