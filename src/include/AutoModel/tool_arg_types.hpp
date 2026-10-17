// Declared parameter types of a request's tools, so a parsed tool call keeps strings as strings.
#pragma once
#include <map>
#include <set>
#include <string>
#include <nlohmann/json.hpp>

using tool_string_args_t = std::map<std::string, std::set<std::string>>;

// Tool name -> the parameters its schema declares as a plain string. Accepts the OpenAI
// {"type":"function","function":{...}} wrapper and the bare {name, parameters} form.
inline tool_string_args_t tool_string_args(const nlohmann::json& tools) {
    tool_string_args_t out;
    if (!tools.is_array()) return out;
    for (const auto& tool : tools) {
        const nlohmann::json& fn = tool.contains("function") ? tool["function"] : tool;
        if (!fn.is_object() || !fn.contains("name") || !fn["name"].is_string()) continue;
        auto& names = out[fn["name"].get<std::string>()];
        if (!fn.contains("parameters") || !fn["parameters"].is_object()) continue;
        const auto& params = fn["parameters"];
        if (!params.contains("properties") || !params["properties"].is_object()) continue;
        for (const auto& [name, schema] : params["properties"].items()) {
            if (!schema.is_object() || !schema.contains("type")) continue;
            const auto& type = schema["type"];
            bool is_string = type.is_string() && type == "string";
            if (type.is_array()) {
                bool has_string = false, has_other = false;
                for (const auto& t : type) {
                    if (t == "string") has_string = true;
                    else if (t != "null") has_other = true;
                }
                is_string = has_string && !has_other;
            }
            if (is_string) names.insert(name);
        }
    }
    return out;
}

// A value is JSON when the schema doesn't say string: "123" stays "123" only if declared a string.
inline nlohmann::json tool_arg_value(const tool_string_args_t& declared, const std::string& tool,
                                     const std::string& param, const std::string& value) {
    auto it = declared.find(tool);
    if (it != declared.end() && it->second.count(param)) return value;
    try {
        return nlohmann::json::parse(value);
    } catch (...) {
        return value;
    }
}
