#include <matplot/matplot.h>
#include <Eigen/Dense>
#include "Solution.h"

using namespace matplot;
#include <matplot/matplot.h>
#include <Eigen/Dense>
#include "Solution.h"

using namespace matplot;

void Solution::visualize() {
    const auto& [boundsX2, boundsX1] = this->f->getOfficialBounds();
    const auto& [x2_min, x2_max] = boundsX2;  // osa Y
    const auto& [x1_min, x1_max] = boundsX1;  // osa X

    int N = 70; // počet bodů na ose
    auto x_lin = linspace(x1_min, x1_max, N);
    auto y_lin = linspace(x2_min, x2_max, N);


    auto [X, Y] = meshgrid(x_lin, y_lin);

    auto Z = transform(X, Y, [this](double x, double y) {
        return this->f->evaluate(Eigen::Vector2d{x, y});
    });

    auto f = figure();
    f->name(this->f->getName() + " " + this->getName());
    auto ax = f->add_axes();
    hold(ax,true);
    const auto& [xLim,yLim] = this->f->getOfficialBounds();
    const auto& [fX,lX] = xLim;
    const auto& [fY,lY] = yLim;
    //ax->xlim({fX,lX});
    //ax->ylim({fY,lY});
    
    auto s = ax->surf(X, Y, Z);
    s->edge_color("gray");

    // Scatter body
    const auto& [Xb,val] = this->bestResult;
    std::vector<double> vx{Xb(0)};
    std::vector<double> vy{Xb(1)};
    std::vector<double> vz{val};

    auto h = ax->scatter3(vx, vy, vz);
    h->marker_style("o");
    h->marker_size(20.0);
    h->marker_color("red");
    h->marker_face_color("red");
    ax->xlabel("X");
    ax->ylabel("Y");
    ax->zlabel("Z");
    ax->grid(true);

      // vykreslí novou figuru s novým surfem a body
}
