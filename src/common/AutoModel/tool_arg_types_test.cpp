// A tool call keeps a parameter a string when the request's schema declares it one.
#include "AutoModel/tool_arg_types.hpp"
#include <cstdio>

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL %s\n", what); ++failures; }
}

int main() {
    auto tools = nlohmann::json::parse(R"([
      {"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{
        "id":{"type":"string"},"count":{"type":"integer"},"flag":{"type":"boolean"},
        "maybe":{"type":["string","null"]},"either":{"type":["string","integer"]},"untyped":{}}}}},
      {"name":"bare","parameters":{"properties":{"code":{"type":"string"}}}}])");
    auto d = tool_string_args(tools);

    check(tool_arg_value(d, "lookup", "id", "123") == "123", "declared string stays a string");
    check(tool_arg_value(d, "lookup", "id", "true") == "true", "declared string 'true' stays a string");
    check(tool_arg_value(d, "lookup", "id", "null").is_string(), "declared string 'null' stays a string");
    check(tool_arg_value(d, "lookup", "maybe", "7").is_string(), "string|null is a string");
    check(tool_arg_value(d, "bare", "code", "42").is_string(), "bare {name, parameters} form");
    check(tool_arg_value(d, "lookup", "count", "5") == 5, "integer parses");
    check(tool_arg_value(d, "lookup", "flag", "true") == true, "boolean parses");
    check(tool_arg_value(d, "lookup", "either", "9") == 9, "string|integer is not forced to a string");
    check(tool_arg_value(d, "lookup", "untyped", "[1,2]").is_array(), "no type parses as JSON");
    check(tool_arg_value(d, "lookup", "id2", "{\"a\":1}").is_object(), "unknown parameter parses as JSON");
    check(tool_arg_value(d, "nosuch", "id", "123") == 123, "unknown tool parses as JSON");
    check(tool_arg_value(d, "lookup", "count", "02134").is_string(), "non-JSON text stays a string");
    check(tool_string_args(nlohmann::json::object()).empty(), "no tools, no entries");
    check(tool_arg_value({}, "lookup", "id", "123") == 123, "no schema: unchanged behavior");

    if (failures) return 1;
    std::printf("tool_arg_types_test passed\n");
    return 0;
}
