#include "sl3dpp/Version.hpp"

#include <planetmodel/version.hpp>

namespace sl3dpp {

std::string_view Version() { return "0.1.0"; }

std::string_view PlanetModelVersion() { return planetmodel::Version(); }

}  // namespace sl3dpp
