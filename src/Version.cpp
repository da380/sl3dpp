#include "sl3dpp/Version.hpp"

// modelfile.hpp is unused here: compiling it in one library translation
// unit is stage 11 WP1's gate, ahead of the io/ layer that will use it.
#include <planetmodel/modelfile.hpp>
#include <planetmodel/version.hpp>

namespace sl3dpp {

std::string_view Version() { return "0.1.0"; }

std::string_view PlanetModelVersion() { return planetmodel::Version(); }

}  // namespace sl3dpp
