#pragma once

#include <string_view>

namespace sl3dpp {

// Version of the sl3dpp library.
std::string_view Version();

// Version of the planetmodel library linked in (its release tag).
std::string_view PlanetModelVersion();

}  // namespace sl3dpp
