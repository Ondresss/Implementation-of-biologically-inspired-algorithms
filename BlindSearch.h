//
// Created by andrew on 21/09/2025.
//

#pragma once
#include "Solution.h"

class BlindSearch : public Solution {
public:
    BlindSearch(int dimension,const std::tuple<int,int>& bounds) : Solution(dimension,bounds) {}
    void run(std::shared_ptr<Function> f, int noIterations) override;
    std::string getName() override { return this->name; }

private:
    std::string name = "Blind Search";
};

