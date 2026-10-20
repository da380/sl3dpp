#pragma once

#include <string>

namespace sl3dpp::io {

/// Why a model-file exchange could not be completed.
struct IoError {
  std::string message;
};

}  // namespace sl3dpp::io
