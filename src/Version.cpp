#include "sl3dpp/Version.hpp"

#include <planetmodel/version.hpp>

namespace sl3dpp {

// SL3DPP_VERSION is the CMake project version (CMakeLists.txt).
std::string_view Version() { return SL3DPP_VERSION; }

std::string_view PlanetModelVersion() { return planetmodel::Version(); }

}  // namespace sl3dpp
