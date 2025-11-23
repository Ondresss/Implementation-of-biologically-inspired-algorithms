#include <numeric>
#include <cmath>
#include "XlsWriter.h"

double XlsWriter::getMean(const std::vector<double> &vals) const {
    if (vals.empty()) return 0.0;
    double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
    return sum / static_cast<double>(vals.size());
}

double XlsWriter::getStdDev(const std::vector<double> &vals) const {
    if (vals.size() < 2) return 0.0;
    double mean = getMean(vals);
    double accum = 0.0;
    for (double v : vals) {
        accum += (v - mean) * (v - mean);
    }
    return std::sqrt(accum / (vals.size() - 1));
}
void XlsWriter::write(int noIterations, int noExperiments) {
    lxw_workbook* workbook = workbook_new("results.xlsx");

    std::vector<std::string> solNames;
    solNames.reserve(this->solutions.size());
    for (const auto& s : this->solutions)
        solNames.push_back(s->getName());

    for (const auto& f : this->testFunctions) {
        lxw_worksheet* sheet = workbook_add_worksheet(workbook, f->getName().c_str());

        worksheet_write_string(sheet, 0, 0, f->getName().c_str(), nullptr);
        worksheet_write_string(sheet, 1, 0," ", nullptr);
        for (int col = 0; col < solNames.size(); ++col)
            worksheet_write_string(sheet, 1, col+1, solNames[col].c_str(), nullptr);

        std::vector<std::vector<double>> results(this->solutions.size());

        for (int exp = 0; exp < noExperiments; ++exp) {
            char BUF[20] = {0};
            std::sprintf(BUF, "exp %d", exp);
            worksheet_write_string(sheet,2 + exp,0,BUF,nullptr);
            for (int col = 0; col < this->solutions.size(); ++col) {

                auto& sol = this->solutions[col];
                sol->run(f, noIterations);
                double bestFitness = sol->getBestSolution();

                worksheet_write_number(sheet, 2 + exp, col+1, bestFitness, nullptr);

                results[col].push_back(bestFitness);
            }
        }
        int meanRow = 2 + noExperiments;
        worksheet_write_string(sheet,meanRow,0,"mean",nullptr);
        for (int col = 0; col < results.size(); ++col)
            worksheet_write_number(sheet, meanRow, col+1, getMean(results[col]), nullptr);

        int stdRow = meanRow + 1;
        worksheet_write_string(sheet,stdRow,0,"stddev",nullptr);
        for (int col = 0; col < results.size(); ++col)
            worksheet_write_number(sheet, stdRow, col+1, getStdDev(results[col]), nullptr);
    }

    workbook_close(workbook);
}
