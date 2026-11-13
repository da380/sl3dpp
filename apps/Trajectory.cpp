#include "Trajectory.hpp"

#include <format>
#include <utility>

#include "Ranks.hpp"
#include "RunHash.hpp"

namespace sl3dpp::app {

TrajectoryBlock ReadTrajectoryBlock(const Json& top, const Run& run,
                                    const std::filesystem::path& directory) {
  auto out = TrajectoryBlock{};
  if (!Has(top, "trajectory")) return out;
  const auto& block = top.At("trajectory");
  if (!block.IsObject()) {
    Fail(block,
         "is not a block: `trajectory` is an object of `directory` and "
         "`record_stride`");
  }
  if (!run.gravitating) {
    Fail(block,
         "records the forward run for the kernels, which a run without "
         "gravity (`gravity`: \"none\") does not have");
  }
  Allow(block, {"directory", "record_stride"});
  if (Has(block, "record_stride")) {
    const auto& member = block.At("record_stride");
    if (!run.times) {
      Fail(member,
           std::format("is a stride through the nodes of a run through time, "
                       "which {}: the static problem's one node is recorded "
                       "whole",
                       kNeedsQuasiStatic));
    }
    out.record_stride = Integer(block, "record_stride");
    if (out.record_stride < 1) Fail(member, "is not a positive stride");
    if (!run.forcing.seismic.empty() && out.record_stride != 1) {
      Fail(member,
           std::format("is {}, and a run under seismic sources "
                       "(`forcing.seismic`) is recorded at every node, its "
                       "states not rebuilt without the sources' forces: give "
                       "1",
                       out.record_stride));
    }
  }
  if (Has(block, "directory")) {
    out.directory = PathOf(block, "directory", directory);
  }
  return out;
}

namespace {

/// A run described for the store, but its processes' layout: one
/// process's, the whole mesh (TrajectoryOf adds a 3-D run's).
io::TrajectoryStore::Description Described(const Run& run, int stride,
                                           std::span<const double> times,
                                           const GRDSolver& solver,
                                           int dimension) {
  auto out = io::TrajectoryStore::Description{};
  out.hash = TrajectoryHashOf(run);
  out.dimension = dimension;
  out.times.assign(times.begin(), times.end());
  out.record_stride = stride;
  out.migrating = run.Migrating();
  out.sources = !run.forcing.seismic.empty();
  out.tide = !run.forcing.tide.empty();
  out.field_size = solver.Grid().FieldSize();
  return out;
}

/// A store's refusal, named at `trajectory.directory`.
template <class F>
auto Named(const Json& top, F&& f) {
  try {
    return f();
  } catch (const io::TrajectoryStoreError& error) {
    Fail(top.At("trajectory").At("directory"), error.what());
  }
}

}  // namespace

io::TrajectoryStore::Description TrajectoryOf(const Run& run, int stride,
                                              std::span<const double> times,
                                              const GRDSolver& solver) {
  return Described(run, stride, times, solver, 1);
}

io::TrajectoryStore::Description TrajectoryOf(const Run& run, int stride,
                                              std::span<const double> times,
                                              const GRDSolver& solver,
                                              const Viscoelastic3D& op) {
  auto out = Described(run, stride, times, solver, 3);
  const auto& ranks = TheRanks();
  out.cuts = op.GetOptions().radial_cuts;
  out.part = io::TrajectoryStore::PartOf(op);
  out.rank = ranks.rank;
  out.ranks = ranks.size;
  out.collectives = ranks.collectives;
  return out;
}

io::TrajectoryStore::Standing SurveyTrajectory(
    const Json& top, const TrajectoryBlock& block,
    const io::TrajectoryStore::Description& run) {
  return Named(top, [&] {
    return io::TrajectoryStore::Survey(block.directory, run);
  });
}

io::TrajectoryStore RecordTrajectory(
    const Json& top, const TrajectoryBlock& block,
    const io::TrajectoryStore::Description& run) {
  return Named(top, [&] {
    return io::TrajectoryStore::Record(block.directory, run);
  });
}

io::TrajectoryStore OpenTrajectory(const Json& top, const TrajectoryBlock& block,
                                   const io::TrajectoryStore::Description& run) {
  return Named(top,
               [&] { return io::TrajectoryStore::Open(block.directory, run); });
}

io::TrajectoryStore ContinueTrajectory(
    const Json& top, const TrajectoryBlock& block,
    const io::TrajectoryStore::Description& run, std::size_t node) {
  try {
    return io::TrajectoryStore::Continue(block.directory, run, node);
  } catch (const io::TrajectoryStoreError& error) {
    Fail(top.At("trajectory").At("directory"),
         std::format("{}; the run resumes from the checkpoint of node {} and "
                     "would continue it. Set \"restart\": false to start the "
                     "run over and record the store whole (it is replaced), "
                     "or drop `trajectory.directory` to resume without "
                     "recording (and remove the store if it is not wanted)",
                     error.what(), node));
  }
}

std::string SizeOf(std::uint64_t bytes) {
  const auto value = static_cast<double>(bytes);
  if (bytes < (std::uint64_t{1} << 10)) return std::format("{} B", bytes);
  if (bytes < (std::uint64_t{1} << 20)) {
    return std::format("{:.1f} KiB", value / 1024.0);
  }
  if (bytes < (std::uint64_t{1} << 30)) {
    return std::format("{:.1f} MiB", value / (1024.0 * 1024.0));
  }
  return std::format("{:.2f} GiB", value / (1024.0 * 1024.0 * 1024.0));
}

}  // namespace sl3dpp::app
