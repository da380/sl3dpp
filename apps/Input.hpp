#pragma once

/**
 * @file Input.hpp
 * @brief The configs the sl3d programs read: planetmodel's strict JSON,
 * with the validation idiom shared across the da380 codes (YSpecpp's):
 * unknown members refused, choices by word, required and optional
 * numbers, and paths resolved against the config file's directory.
 */

#include <filesystem>
#include <format>
#include <initializer_list>
#include <planetmodel/json.hpp>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace sl3dpp::app {

using planetmodel::Json;

/// Throws an error naming the file, the member and the fault.
[[noreturn]] void Fail(const Json& json, std::string_view message);

/// Refuses any member of an object not among those named, so that a
/// misspelt member is not silently ignored.
void Allow(const Json& json, std::span<const std::string_view> names);
void Allow(const Json& json, std::initializer_list<std::string_view> names);

/// Whether an object has a member, which `At` would otherwise refuse.
[[nodiscard]] bool Has(const Json& object, std::string_view key);

/// One of a set of words, each standing for a value.
template <class T>
T Choice(const Json& json,
         std::initializer_list<std::pair<std::string_view, T>> choices) {
  const auto& word = json.AsString();
  for (const auto& [name, value] : choices) {
    if (word == name) return value;
  }
  auto names = std::string{};
  for (const auto& [name, value] : choices) {
    names += (names.empty() ? "\"" : ", \"") + std::string(name) + "\"";
  }
  Fail(json, std::format("is \"{}\", which is none of {}", word, names));
}

/// A number that must be there, and one that may not be.
[[nodiscard]] double Number(const Json& object, std::string_view key);
[[nodiscard]] double Number(const Json& object, std::string_view key,
                            double fallback);

/// An integer that must be there, and one that may not be.
[[nodiscard]] int Integer(const Json& object, std::string_view key);
[[nodiscard]] int Integer(const Json& object, std::string_view key,
                          int fallback);

/// A string that must be there, and one that may not be.
[[nodiscard]] std::string String(const Json& object, std::string_view key);
[[nodiscard]] std::string String(const Json& object, std::string_view key,
                                 std::string_view fallback);

/// A path member, a relative one resolved against the config file's
/// directory.
[[nodiscard]] std::filesystem::path PathOf(
    const Json& object, std::string_view key,
    const std::filesystem::path& directory);

/// The writing time in UTC, ISO 8601: the stamp of the programs'
/// provenance headers.
[[nodiscard]] std::string CreatedNow();

}  // namespace sl3dpp::app
