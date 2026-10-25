// The radially substructured degree solves on the real systems, step 1 of
// docs/dev/radial-substructuring.md: every spheroidal and toroidal degree
// system of a model at lmax, factorised and solved directly and with the
// mesh cut into P segments (RadialMesh::BalancedCuts), 2(l + 1) columns
// per degree as a 3D pass solves them. One thread throughout. Per P:
//
//   build     the factorisations summed over degrees (the direct one, or
//             the segments', the coupling columns and the separators')
//   solve     the solves summed over degrees, best of `repeats` (serial:
//             the total work a distributed solve shares out)
//   balance   sum_l (segment dofs) / (P max_l segment dofs) weighted by
//             the columns: the share of the work a rank of the largest
//             segment carries, against a perfect split (1 is perfect)
//   at lmax   the segments the highest degree keeps (the deep ones drop
//             out above the truncation radius)
//   agree     max over degrees and columns of |x - x_direct| / |x_direct|
//
// Usage:
//   substructure_timing <deck> <lmax> [repeats] [parts...]
//
// The deck is read as step_timing reads it (the old sl3d format, e.g.
// tests/data/prem.simp.vm1); parts default to 2 4 8 16.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/DegreeOperator.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/ToroidalOperator.hpp"

using namespace sl3dpp;

namespace {

double Seconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

double RelativeDifference(std::span<const double> got,
                          std::span<const double> want) {
  auto scale = 0.0, diff = 0.0;
  for (auto i = std::size_t{0}; i < want.size(); ++i) {
    scale = std::max(scale, std::abs(want[i]));
    diff = std::max(diff, std::abs(got[i] - want[i]));
  }
  return diff / scale;
}

struct Totals {
  double build = 0.0;
  double solve = 0.0;
  double work = 0.0;     // sum of dofs x columns
  double largest = 0.0;  // sum of the largest segment's dofs x columns
  int top = 0;           // segments at lmax
  double agree = 0.0;
};

/// The dof counts of the segments a degree's cuts make, from the first dof
/// of each cut node above the degree's first element.
template <typename System, typename FirstDof>
std::vector<int> SegmentSizes(const System& system, const RadialMesh& mesh,
                              std::span<const int> cuts, FirstDof&& firstDof) {
  auto bounds = std::vector<int>{0};
  for (const auto cut : cuts) {
    if (cut <= system.FirstElement()) continue;
    if (const auto d = firstDof(mesh.GlobalIndex(cut, 0)); d >= 0) {
      bounds.push_back(d);
    }
  }
  bounds.push_back(system.DofCount());
  auto out = std::vector<int>{};
  for (auto k = std::size_t{1}; k < bounds.size(); ++k) {
    out.push_back(bounds[k] - bounds[k - 1]);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: substructure_timing <deck> <lmax> [repeats] "
                 "[parts...]\n");
    return 2;
  }
  const auto lmax = std::atoi(argv[2]);
  const auto repeats = argc > 3 ? std::atoi(argv[3]) : 3;
  auto partsList = std::vector<int>{};
  for (auto k = 4; k < argc; ++k) partsList.push_back(std::atoi(argv[k]));
  if (partsList.empty()) partsList = {2, 4, 8, 16};

  auto deck = Deck::ReadLegacySl3d(argv[1]).value();
  auto model =
      EarthModel::FromDeck(deck, {.spline = SplineKind::Natural}).value();
  auto mesh = RadialMesh::CreateGraded(model, lmax).value();
  auto material = Material::Create(model, mesh).value();
  std::printf("# %s, lmax %d: %d elements of %d GLL nodes; one thread\n",
              argv[1], lmax, mesh.ElementCount(), mesh.NGll());

  auto engine = std::mt19937{11};
  auto uniform = std::uniform_real_distribution<double>{-1.0, 1.0};

  // The direct solves: the reference and the baseline times.
  auto direct = Totals{};
  auto rhs = std::vector<std::vector<double>>{};
  auto want = std::vector<std::vector<double>>{};
  auto rhsT = std::vector<std::vector<double>>{};
  auto wantT = std::vector<std::vector<double>>{};
  for (auto l = 0; l <= lmax; ++l) {
    const auto columns = 2 * (l + 1);
    auto start = std::chrono::steady_clock::now();
    const DegreeOperator system(material, l);
    direct.build += Seconds(start);
    auto b = std::vector<double>(static_cast<std::size_t>(system.DofCount()) *
                                 columns);
    for (auto& v : b) v = uniform(engine);
    auto best = 1e300;
    auto x = std::vector<double>{};
    for (auto r = 0; r < repeats; ++r) {
      start = std::chrono::steady_clock::now();
      x = system.Solve(b, columns);
      best = std::min(best, Seconds(start));
    }
    direct.solve += best;
    rhs.push_back(std::move(b));
    want.push_back(std::move(x));
    if (l == 0) continue;
    start = std::chrono::steady_clock::now();
    const ToroidalOperator toroidal(material, l);
    direct.build += Seconds(start);
    auto bt = toroidal.Apply(
        std::vector<double>(
            static_cast<std::size_t>(toroidal.DofCount()) * columns, 1.0),
        columns);
    best = 1e300;
    for (auto r = 0; r < repeats; ++r) {
      start = std::chrono::steady_clock::now();
      x = toroidal.Solve(bt, columns);
      best = std::min(best, Seconds(start));
    }
    direct.solve += best;
    rhsT.push_back(std::move(bt));
    wantT.push_back(std::move(x));
  }
  std::printf(
      "# parts  build_s  solve_s  solve/direct  balance  at_lmax  agree\n");
  std::printf("%7d  %7.3f  %7.3f  %12.2f  %7.2f  %7d  %.1e\n", 1, direct.build,
              direct.solve, 1.0, 1.0, 1, 0.0);

  for (const auto parts : partsList) {
    const auto cuts = mesh.BalancedCuts(parts);
    auto cut = Totals{};
    for (auto l = 0; l <= lmax; ++l) {
      const auto columns = 2 * (l + 1);
      auto start = std::chrono::steady_clock::now();
      const DegreeOperator system(material, l, {.cuts = cuts});
      cut.build += Seconds(start);
      auto best = 1e300;
      auto x = std::vector<double>{};
      for (auto r = 0; r < repeats; ++r) {
        start = std::chrono::steady_clock::now();
        x = system.Solve(rhs[l], columns);
        best = std::min(best, Seconds(start));
      }
      cut.solve += best;
      cut.agree = std::max(cut.agree, RelativeDifference(x, want[l]));
      const auto sizes = SegmentSizes(system, mesh, cuts, [&](int node) {
        for (const auto c : {Component::U, Component::Phi}) {
          if (const auto d = system.Dof(node, c); d >= 0) return d;
        }
        return -1;
      });
      for (const auto s : sizes) cut.work += 1.0 * s * columns;
      cut.largest += 1.0 * *std::ranges::max_element(sizes) * columns;
      if (l == lmax) cut.top = system.SegmentCount();
      if (l == 0) continue;
      start = std::chrono::steady_clock::now();
      const ToroidalOperator toroidal(material, l, {.cuts = cuts});
      cut.build += Seconds(start);
      best = 1e300;
      for (auto r = 0; r < repeats; ++r) {
        start = std::chrono::steady_clock::now();
        x = toroidal.Solve(rhsT[l - 1], columns);
        best = std::min(best, Seconds(start));
      }
      cut.solve += best;
      cut.agree = std::max(cut.agree, RelativeDifference(x, wantT[l - 1]));
    }
    const auto balance = cut.work / (parts * cut.largest);
    std::printf("%7d  %7.3f  %7.3f  %12.2f  %7.2f  %7d  %.1e\n", parts,
                cut.build, cut.solve, cut.solve / direct.solve, balance,
                cut.top, cut.agree);
  }
  return 0;
}
