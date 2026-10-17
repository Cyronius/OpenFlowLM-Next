// Traces: OPEN-TTS-G2P (specs/open-tts/spec.md). misaki's own output on a lexicon excerpt; no device, no model.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "g2p_en.hpp"

using namespace open_tts;

static int failures = 0;

static void expect_eq(const std::string& what, const std::string& got, const std::string& want) {
    const bool ok = got == want;
    if (!ok) ++failures;
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) std::printf("     got  [%s]\n     want [%s]\n", got.c_str(), want.c_str());
}

int main(int argc, char** argv) {
    const std::filesystem::path dir = argc > 1 ? argv[1] : "testdata";
    expect_eq("cardinal 1234", num2words_cardinal(1234), "one thousand, two hundred and thirty-four");
    expect_eq("cardinal 2500000", num2words_cardinal(2500000), "two million, five hundred thousand");
    expect_eq("cardinal 1001", num2words_cardinal(1001), "one thousand and one");
    expect_eq("ordinal 21", num2words_ordinal(21), "twenty-first");
    expect_eq("ordinal 100", num2words_ordinal(100), "one hundredth");
    expect_eq("ordinal 20", num2words_ordinal(20), "twentieth");
    expect_eq("year 1984", num2words_year(1984), "nineteen eighty-four");
    expect_eq("year 1905", num2words_year(1905), "nineteen oh-five");
    expect_eq("year 2001", num2words_year(2001), "two thousand and one");
    expect_eq("year 2010", num2words_year(2010), "twenty ten");

    const std::string gold = (dir / "g2p_gold.json").string(), silver = (dir / "g2p_silver.json").string();
    std::ifstream in(dir / "g2p_cases.json", std::ios::binary);
    if (!in) {
        std::printf("FAIL fixtures missing in %s (run testdata/gen_g2p_fixtures.py)\n", dir.string().c_str());
        return 1;
    }
    const auto fx = nlohmann::json::parse(in);
    EnglishG2P skip(gold, silver, false, false);
    for (const auto& c : fx["cases"])
        expect_eq("misaki: " + c["text"].get<std::string>(), skip(c["text"].get<std::string>()).phonemes,
                  c["phonemes"].get<std::string>());

    // Spell mode reads an unknown word the way misaki's get_NNP reads an acronym.
    EnglishG2P spell(gold, silver, false, true);
    for (const auto& [word, ps] : fx["spell"].items()) {
        const G2PResult r = spell(word);
        expect_eq("unknown word reported: " + word, r.unknown.empty() ? "" : r.unknown[0], word);
        expect_eq("unknown word spelled: " + word, r.phonemes, ps.get<std::string>());
        expect_eq("unknown word dropped in skip mode: " + word, skip(word).phonemes, "");
    }
    std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
