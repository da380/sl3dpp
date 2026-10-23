#include "Input.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <planetmodel/error.hpp>

namespace sl3dpp::app {

void Fail(const Json& json, std::string_view message) {
  throw planetmodel::Error(json.Source(), json.Path(), message);
}

void Allow(const Json& json, std::span<const std::string_view> names) {
  for (const auto& [key, value] : json.AsObject()) {
    if (std::ranges::find(names, key) == names.end()) {
      Fail(value, "is not a recognised member");
    }
  }
}

void Allow(const Json& json, std::initializer_list<std::string_view> names) {
  Allow(json, std::span(names.begin(), names.size()));
}

bool Has(const Json& object, std::string_view key) {
  return std::ranges::any_of(object.AsObject(), [&key](const auto& member) {
    return member.first == key;
  });
}

double Number(const Json& object, std::string_view key) {
  return object.At(key).AsNumber();
}

double Number(const Json& object, std::string_view key, double fallback) {
  return Has(object, key) ? Number(object, key) : fallback;
}

int Integer(const Json& object, std::string_view key) {
  const auto value = object.At(key).AsNumber();
  if (value != std::floor(value)) Fail(object.At(key), "is not an integer");
  return static_cast<int>(value);
}

int Integer(const Json& object, std::string_view key, int fallback) {
  return Has(object, key) ? Integer(object, key) : fallback;
}

std::string String(const Json& object, std::string_view key) {
  return object.At(key).AsString();
}

std::string String(const Json& object, std::string_view key,
                   std::string_view fallback) {
  return Has(object, key) ? String(object, key) : std::string{fallback};
}

std::filesystem::path PathOf(const Json& object, std::string_view key,
                             const std::filesystem::path& directory) {
  const auto path = std::filesystem::path{object.At(key).AsString()};
  return path.is_absolute() ? path : directory / path;
}

std::string CreatedNow() {
  const auto now = std::chrono::floor<std::chrono::seconds>(
      std::chrono::system_clock::now());
  return std::format("{:%FT%T}+00:00", now);
}

}  // namespace sl3dpp::app
