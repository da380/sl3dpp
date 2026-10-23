#include "sl3dpp/io/FieldHistory.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <limits>
#include <map>
#include <planetmodel/rules.hpp>
#include <planetmodel/surfacefile.hpp>
#include <planetmodel/units.hpp>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sl3dpp::io {

namespace {

namespace pm = planetmodel;

/// The lateral rule a variable states, the harmonic one read as cubic
/// (this boundary evaluates no harmonic rule).
std::expected<pm::LateralRule, IoError> RuleOf(const pm::FileVariable& var,
                                               const std::string& source) {
  const auto text = var.Text("lateral_interpolation");
  if (!text) {
    return std::unexpected(IoError{std::format(
        "{}: {} states no lateral_interpolation", source, var.name)});
  }
  if (*text == "linear") return pm::LateralRule::Linear;
  if (*text == "cubic" || *text == "harmonic") return pm::LateralRule::Cubic;
  return std::unexpected(IoError{std::format(
      "{}: {} states the lateral rule {}, which is not read here", source,
      var.name, *text)});
}

/// The factor one stored unit of a variable is in SI: its declared
/// dimensions under the file's scales, one where it declares none.
double FactorOf(const pm::FileVariable& var, const pm::Scales& scales) {
  const auto exps = var.Numbers("dimensions");
  if (!exps || exps->size() != 4) return 1.0;
  return scales.Factor(pm::Dimensions{.mass = static_cast<int>((*exps)[0]),
                                      .length = static_cast<int>((*exps)[1]),
                                      .time = static_cast<int>((*exps)[2]),
                                      .temperature =
                                          static_cast<int>((*exps)[3])});
}

}  // namespace

struct FieldHistory::Impl {
  pm::SurfaceHistory history;
  std::vector<std::string> names;
  std::optional<std::pair<double, double>> span;
  std::string source;
  pm::Extrapolation extrapolation = pm::Extrapolation::Hold;
};

std::expected<FieldHistory, IoError> FieldHistory::Read(
    const std::filesystem::path& path, const SurfaceGrid& grid,
    const FieldHistoryOptions& options) {
  try {
    const auto file = pm::ReadSurfaceFile(path);
    const auto& from = *file.grid;
    const auto src_size = from.NTheta() * from.NPhi();

    // the solver grid as planetmodel sees it, and its nodes to evaluate at
    auto resampled = pm::SurfaceFile{};
    resampled.grid = pm::AngularGrid(
        std::vector<double>(grid.CoLatitudes().begin(),
                            grid.CoLatitudes().end()),
        std::vector<double>(grid.Longitudes().begin(),
                            grid.Longitudes().end()));
    const auto ntheta = static_cast<std::size_t>(grid.NumberOfCoLatitudes());
    const auto nphi = static_cast<std::size_t>(grid.NumberOfLongitudes());

    // the axes in SI seconds; everything downstream is SI
    const auto second = file.scales.Time();
    for (const auto& [axis, held] : {std::pair{&file.time, &resampled.time},
                                     std::pair{&file.instant,
                                               &resampled.instant}}) {
      if (!*axis) continue;
      auto scaled = **axis;
      for (auto& t : scaled) t *= second;
      *held = std::move(scaled);
    }
    resampled.source = file.source;
    resampled.created = file.created;
    resampled.title = file.title;
    resampled.meta = file.meta;
    resampled.citation = file.citation;

    auto names = std::vector<std::string>{};
    for (const auto& [name, var] : file.variables) {
      if (name.ends_with("_poles")) continue;
      const auto rule = RuleOf(var, path.string());
      if (!rule) return std::unexpected(rule.error());
      const auto factor = FactorOf(var, file.scales);
      const auto rows = var.values.size() / src_size;
      const auto* poles_var = [&]() -> const pm::FileVariable* {
        const auto it = file.variables.find(name + "_poles");
        return it == file.variables.end() ? nullptr : &it->second;
      }();
      auto values = std::vector<double>{};
      values.reserve(rows * ntheta * nphi);
      for (auto k = std::size_t{0}; k < rows; ++k) {
        const auto slice =
            std::span{var.values}.subspan(k * src_size, src_size);
        // the file's polar values; a missing pole on a grid lacking it is
        // taken as the mean of its nearest row, the local estimate this
        // boundary can make without harmonic machinery
        auto estimated = std::array<double, 2>{};
        auto poles = std::span<const double>{};
        if (poles_var) {
          poles = std::span{poles_var->values}.subspan(2 * k, 2);
        } else if (from.LacksPole()) {
          const auto nan = std::numeric_limits<double>::quiet_NaN();
          auto mean = [&slice, src_nphi = from.NPhi()](std::size_t row) {
            auto sum = 0.0;
            for (auto j = std::size_t{0}; j < src_nphi; ++j) {
              sum += slice[row * src_nphi + j];
            }
            return sum / static_cast<double>(src_nphi);
          };
          estimated = {from.NorthPoleNode() ? nan : mean(0),
                       from.SouthPoleNode() ? nan : mean(from.NTheta() - 1)};
          poles = estimated;
        }
        const auto f = pm::LateralInterpolant(from, slice, poles, *rule, name);
        for (auto i = std::size_t{0}; i < ntheta; ++i) {
          const auto theta = grid.CoLatitudes()[i];
          for (auto j = std::size_t{0}; j < nphi; ++j) {
            values.push_back(factor * f.Value(theta, grid.Longitudes()[j]));
          }
        }
      }
      auto out = var;
      out.values = std::move(values);
      out.dimensions.clear();
      out.shape.clear();
      resampled.variables.emplace(name, std::move(out));
      names.push_back(name);
    }
    pm::ValidateSurfaceFile(resampled, path.string());

    auto impl = std::make_shared<Impl>(Impl{
        .history = pm::SurfaceHistory(std::move(resampled)),
        .names = std::move(names),
        .span = {},
        .source = file.source,
        .extrapolation = options.hold_outside ? pm::Extrapolation::Hold
                                              : pm::Extrapolation::Refuse});
    if (const auto& time = impl->history.File().time) {
      impl->span = std::pair{time->front(), time->back()};
    }
    return FieldHistory(std::move(impl));
  } catch (const pm::Error& error) {
    return std::unexpected(IoError{error.what()});
  }
}

const std::vector<std::string>& FieldHistory::Names() const {
  return impl_->names;
}

bool FieldHistory::Has(std::string_view name) const {
  return std::ranges::find(impl_->names, name) != impl_->names.end();
}

std::optional<std::pair<double, double>> FieldHistory::Span() const {
  return impl_->span;
}

std::vector<double> FieldHistory::Values(std::string_view name,
                                         double time) const {
  try {
    return impl_->history.Values(name, time, impl_->extrapolation);
  } catch (const pm::Error& error) {
    throw std::runtime_error(error.what());
  }
}

const std::string& FieldHistory::Source() const { return impl_->source; }

}  // namespace sl3dpp::io
