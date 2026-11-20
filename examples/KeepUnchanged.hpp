#pragma once

/**
 * @file examples/KeepUnchanged.hpp
 * @brief A generated input replaced only when its content changes. The
 * build's generators (MakeSyntheticEarth, MakeSyntheticModuli,
 * MakeSyntheticPerturbation) write their
 * netCDF file beside its target and then keep the target as it is — its
 * bytes, and so the hash a run takes of its input files — where the two
 * hold the same dimensions, variables, values and attributes, the root
 * group's creation stamp `created` aside; otherwise the new file replaces
 * it. A run's checkpoints and trajectory stores then stay its own across
 * rebuilds that leave the inputs as they were. The Python generators do
 * the same (python/inputs.py, `keep_unchanged`).
 */

#include <netcdf.h>

#include <cstddef>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <system_error>
#include <vector>

namespace examples {

namespace keep {

/// Closes a netCDF file on leaving scope.
struct File {
  int id = -1;
  explicit File(const std::filesystem::path& path) {
    if (nc_open(path.c_str(), NC_NOWRITE, &id) != NC_NOERR) id = -1;
  }
  ~File() {
    if (id >= 0) nc_close(id);
  }
  File(const File&) = delete;
  File& operator=(const File&) = delete;
};

/// The bytes of a fixed-size attribute or variable, or its strings joined
/// with their lengths (NC_STRING); empty and false for a user-defined type.
inline bool ValuesOf(int group, int variable, const char* attribute,
                     nc_type type, std::size_t count, std::string& out) {
  out.clear();
  if (type >= NC_FIRSTUSERTYPEID) return false;
  if (type == NC_STRING) {
    auto strings = std::vector<char*>(count, nullptr);
    const auto status =
        attribute != nullptr
            ? nc_get_att_string(group, variable, attribute, strings.data())
            : (count == 0 ? NC_NOERR
                          : nc_get_var_string(group, variable, strings.data()));
    if (status != NC_NOERR) return false;
    for (auto* s : strings) {
      const auto text = std::string{s != nullptr ? s : ""};
      out += std::to_string(text.size()) + ':' + text;
    }
    if (count > 0) nc_free_string(count, strings.data());
    return true;
  }
  auto size = std::size_t{0};
  if (nc_inq_type(group, type, nullptr, &size) != NC_NOERR) return false;
  out.resize(size * count);
  if (count == 0) return true;
  const auto status = attribute != nullptr
                          ? nc_get_att(group, variable, attribute, out.data())
                          : nc_get_var(group, variable, out.data());
  return status == NC_NOERR;
}

/// The attributes of a variable (or of the group, NC_GLOBAL) by name: type,
/// count and values; the root group's `created` left out.
inline bool AttributesOf(int group, int variable, bool root,
                         std::map<std::string, std::string>& out) {
  out.clear();
  auto count = 0;
  if (nc_inq_varnatts(group, variable, &count) != NC_NOERR) return false;
  for (auto i = 0; i < count; ++i) {
    char name[NC_MAX_NAME + 1] = {};
    if (nc_inq_attname(group, variable, i, name) != NC_NOERR) return false;
    if (root && variable == NC_GLOBAL && std::strcmp(name, "created") == 0) continue;
    nc_type type{};
    auto length = std::size_t{0};
    if (nc_inq_att(group, variable, name, &type, &length) != NC_NOERR) return false;
    auto values = std::string{};
    if (!ValuesOf(group, variable, name, type, length, values)) return false;
    out[name] = std::to_string(type) + '/' + std::to_string(length) + '/' + values;
  }
  return true;
}

/// A group's dimensions, variables (their types, dimensions, attributes and
/// values), attributes and subgroups, in one description per item.
inline bool Describe(int group, bool root, std::map<std::string, std::string>& out) {
  out.clear();
  auto attributes = std::map<std::string, std::string>{};
  if (!AttributesOf(group, NC_GLOBAL, root, attributes)) return false;
  for (const auto& [name, value] : attributes) out["attribute " + name] = value;
  auto dims = 0;
  if (nc_inq_ndims(group, &dims) != NC_NOERR) return false;
  auto dimIds = std::vector<int>(static_cast<std::size_t>(dims));
  if (dims > 0 && nc_inq_dimids(group, &dims, dimIds.data(), 0) != NC_NOERR) {
    return false;
  }
  for (const auto id : dimIds) {
    char name[NC_MAX_NAME + 1] = {};
    auto length = std::size_t{0};
    if (nc_inq_dim(group, id, name, &length) != NC_NOERR) return false;
    out[std::string{"dimension "} + name] = std::to_string(length);
  }
  auto variables = 0;
  if (nc_inq_nvars(group, &variables) != NC_NOERR) return false;
  for (auto v = 0; v < variables; ++v) {
    char name[NC_MAX_NAME + 1] = {};
    nc_type type{};
    auto ndims = 0;
    if (nc_inq_var(group, v, name, &type, &ndims, nullptr, nullptr) != NC_NOERR) {
      return false;
    }
    auto ids = std::vector<int>(static_cast<std::size_t>(ndims));
    if (ndims > 0 && nc_inq_vardimid(group, v, ids.data()) != NC_NOERR) return false;
    auto shape = std::string{};
    auto count = std::size_t{1};
    for (const auto id : ids) {
      char dim[NC_MAX_NAME + 1] = {};
      auto length = std::size_t{0};
      if (nc_inq_dim(group, id, dim, &length) != NC_NOERR) return false;
      shape += std::string{dim} + '=' + std::to_string(length) + ',';
      count *= length;
    }
    auto values = std::string{};
    if (!ValuesOf(group, v, nullptr, type, count, values)) return false;
    auto held = std::map<std::string, std::string>{};
    if (!AttributesOf(group, v, false, held)) return false;
    auto description = std::to_string(type) + '/' + shape + '/' + values;
    for (const auto& [key, value] : held) description += '|' + key + '=' + value;
    out[std::string{"variable "} + name] = std::move(description);
  }
  auto groups = 0;
  if (nc_inq_grps(group, &groups, nullptr) != NC_NOERR) return false;
  auto groupIds = std::vector<int>(static_cast<std::size_t>(groups));
  if (groups > 0 && nc_inq_grps(group, &groups, groupIds.data()) != NC_NOERR) {
    return false;
  }
  for (const auto id : groupIds) {
    char name[NC_MAX_NAME + 1] = {};
    if (nc_inq_grpname(id, name) != NC_NOERR) return false;
    auto inner = std::map<std::string, std::string>{};
    if (!Describe(id, false, inner)) return false;
    for (const auto& [key, value] : inner) out[std::string{name} + "/" + key] = value;
  }
  return true;
}

}  // namespace keep

/// Whether two netCDF files hold the same content but for the root group's
/// `created` stamp (false where either cannot be read, or holds a type the
/// comparison does not read).
inline bool SameButCreated(const std::filesystem::path& a,
                           const std::filesystem::path& b) {
  const keep::File first(a), second(b);
  if (first.id < 0 || second.id < 0) return false;
  auto left = std::map<std::string, std::string>{};
  auto right = std::map<std::string, std::string>{};
  return keep::Describe(first.id, true, left) &&
         keep::Describe(second.id, true, right) && left == right;
}

/// Moves `written` onto `target`, unless `target` holds the same content
/// (SameButCreated): then `written` is removed and `target` left as it is,
/// its bytes untouched. Returns whether `target` was replaced.
inline bool ReplaceIfChanged(const std::filesystem::path& written,
                             const std::filesystem::path& target) {
  if (std::filesystem::exists(target) && SameButCreated(written, target)) {
    std::filesystem::remove(written);
    return false;
  }
  std::filesystem::rename(written, target);
  return true;
}

/// The file a generator writes before ReplaceIfChanged: beside the target,
/// `<stem>.new<extension>`.
inline std::filesystem::path Beside(const std::filesystem::path& target) {
  auto name = target.stem();
  name += ".new";
  name += target.extension();
  return target.parent_path() / name;
}

}  // namespace examples
