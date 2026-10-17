// English G2P for Kokoro: misaki 0.9.4's en.py dictionary path, spaCy replaced by fixed tags (2026-10-16).
#include "g2p_en.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cwctype>
#include <fstream>
#include <functional>
#include <set>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace open_tts {

std::u32string utf8_to_u32(const std::string& s) {
    std::u32string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        char32_t cp;
        int n;
        if (c < 0x80) { cp = c; n = 1; }
        else if ((c >> 5) == 6) { cp = c & 0x1F; n = 2; }
        else if ((c >> 4) == 14) { cp = c & 0x0F; n = 3; }
        else if ((c >> 3) == 30) { cp = c & 0x07; n = 4; }
        else { ++i; continue; }
        if (i + n > s.size()) break;
        for (int k = 1; k < n; ++k) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
        out.push_back(cp);
        i += n;
    }
    return out;
}

std::string u32_to_utf8(const std::u32string& s) {
    std::string out;
    for (char32_t c : s) {
        if (c < 0x80) out.push_back((char)c);
        else if (c < 0x800) { out.push_back((char)(0xC0 | (c >> 6))); out.push_back((char)(0x80 | (c & 0x3F))); }
        else if (c < 0x10000) {
            out.push_back((char)(0xE0 | (c >> 12)));
            out.push_back((char)(0x80 | ((c >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (c & 0x3F)));
        } else {
            out.push_back((char)(0xF0 | (c >> 18)));
            out.push_back((char)(0x80 | ((c >> 12) & 0x3F)));
            out.push_back((char)(0x80 | ((c >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (c & 0x3F)));
        }
    }
    return out;
}

namespace {

using u32 = std::u32string;

u32 U(const char* s) { return utf8_to_u32(s); }

bool is_letter(char32_t c) {
    if (c < 0x80) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    return c < 0x10000 && std::iswalpha((wint_t)c);
}
bool is_upper(char32_t c) {
    if (c < 0x80) return c >= 'A' && c <= 'Z';
    return c < 0x10000 && std::iswupper((wint_t)c);
}
bool is_lower(char32_t c) {
    if (c < 0x80) return c >= 'a' && c <= 'z';
    return c < 0x10000 && std::iswlower((wint_t)c);
}
bool is_digit(char32_t c) { return c >= '0' && c <= '9'; }
bool is_space(char32_t c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0xA0 || c == 0x3000; }
char32_t lower_c(char32_t c) {
    if (c < 0x80) return (c >= 'A' && c <= 'Z') ? c + 32 : c;
    return c < 0x10000 ? (char32_t)std::towlower((wint_t)c) : c;
}
char32_t upper_c(char32_t c) {
    if (c < 0x80) return (c >= 'a' && c <= 'z') ? c - 32 : c;
    return c < 0x10000 ? (char32_t)std::towupper((wint_t)c) : c;
}
// Latin-1 and Latin Extended-A letters (U+00C0..U+017F) without their accents, for spelling unknown words.
const char* const LATIN_FOLD[0x180 - 0xC0] = {
    "A", "A", "A", "A", "A", "A", "AE", "C", "E", "E", "E", "E", "I", "I", "I", "I",
    "D", "N", "O", "O", "O", "O", "O", "", "O", "U", "U", "U", "U", "Y", "TH", "SS",
    "A", "A", "A", "A", "A", "A", "AE", "C", "E", "E", "E", "E", "I", "I", "I", "I",
    "D", "N", "O", "O", "O", "O", "O", "", "O", "U", "U", "U", "U", "Y", "TH", "Y",
    "A", "A", "A", "A", "A", "A", "C", "C", "C", "C", "C", "C", "C", "C", "D", "D",
    "D", "D", "E", "E", "E", "E", "E", "E", "E", "E", "E", "E", "G", "G", "G", "G",
    "G", "G", "G", "G", "H", "H", "H", "H", "I", "I", "I", "I", "I", "I", "I", "I",
    "I", "I", "IJ", "IJ", "J", "J", "K", "K", "K", "L", "L", "L", "L", "L", "L", "L",
    "L", "L", "L", "N", "N", "N", "N", "N", "N", "N", "N", "N", "O", "O", "O", "O",
    "O", "O", "OE", "OE", "R", "R", "R", "R", "R", "R", "S", "S", "S", "S", "S", "S",
    "S", "S", "T", "T", "T", "T", "T", "T", "U", "U", "U", "U", "U", "U", "U", "U",
    "U", "U", "U", "U", "W", "W", "Y", "Y", "Y", "Z", "Z", "Z", "Z", "Z", "Z", "S",
};
u32 fold_latin(char32_t c) {
    if (c >= 0xC0 && c < 0x180) return utf8_to_u32(LATIN_FOLD[c - 0xC0]);
    return u32(1, upper_c(c));
}
u32 lower(const u32& s) { u32 r = s; for (auto& c : r) c = lower_c(c); return r; }
u32 upper(const u32& s) { u32 r = s; for (auto& c : r) c = upper_c(c); return r; }
u32 capitalize(const u32& s) {
    u32 r = lower(s);
    if (!r.empty()) r[0] = upper_c(r[0]);
    return r;
}
bool isalpha_str(const u32& s) {
    if (s.empty()) return false;
    for (char32_t c : s) if (!is_letter(c)) return false;
    return true;
}
bool all_digits(const u32& s) {
    if (s.empty()) return false;
    for (char32_t c : s) if (!is_digit(c)) return false;
    return true;
}
bool contains(const u32& set, char32_t c) { return set.find(c) != u32::npos; }
bool contains_str(const u32& s, const u32& sub) { return s.find(sub) != u32::npos; }
bool ends_with(const u32& s, const u32& suf) { return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0; }
bool ends_with(const u32& s, const char* suf) { return ends_with(s, U(suf)); }
bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }
u32 replace_all(u32 s, const u32& from, const u32& to) {
    if (from.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != u32::npos) { s.replace(pos, from.size(), to); pos += to.size(); }
    return s;
}

const char32_t PRIMARY = 0x02C8, SECONDARY = 0x02CC;
const u32 STRESSES = U("ˌˈ");
const u32 VOWELS = U("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ");
const u32 CONSONANTS = U("bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ");
const u32 US_TAUS = U("AIOWYiuæɑəɛɪɹʊʌ");
const u32 DIPHTHONGS = U("AIOQWYʤʧ");
const u32 PUNCTS = U(";:,.!?—…\"“”");
const u32 NON_QUOTE_PUNCTS = U(";:,.!?—…");
const u32 SUBTOKEN_JUNKS = U("',-._‘’/");
const u32 QUOTES = U("'‘’");
const std::set<std::string> PUNCT_TAGS = {".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP"};

bool lexicon_ords(const u32& w) {
    for (char32_t c : w)
        if (!(c == 39 || c == 45 || (c >= 65 && c <= 90) || (c >= 97 && c <= 122))) return false;
    return true;
}

using Stress = std::optional<double>;

u32 apply_stress(const u32& ps, Stress stress) {
    auto has_any = [&](const u32& set) { for (char32_t c : ps) if (contains(set, c)) return true; return false; };
    auto restress = [&](const u32& p) {
        std::vector<std::pair<double, char32_t>> ips;
        for (size_t i = 0; i < p.size(); ++i) ips.push_back({(double)i, p[i]});
        for (size_t i = 0; i < p.size(); ++i)
            if (contains(STRESSES, p[i]))
                for (size_t j = i; j < p.size(); ++j)
                    if (contains(VOWELS, p[j])) { ips[i].first = (double)j - 0.5; break; }
        std::stable_sort(ips.begin(), ips.end());
        u32 r;
        for (auto& [k, c] : ips) r.push_back(c);
        return r;
    };
    if (!stress) return ps;
    const double s = *stress;
    if (s < -1) {
        u32 r;
        for (char32_t c : ps) if (c != PRIMARY && c != SECONDARY) r.push_back(c);
        return r;
    }
    const bool has_primary = ps.find(PRIMARY) != u32::npos;
    const bool has_secondary = ps.find(SECONDARY) != u32::npos;
    if (s == -1 || ((s == 0 || s == -0.5) && has_primary)) {
        u32 r;
        for (char32_t c : ps) {
            if (c == SECONDARY) continue;
            r.push_back(c == PRIMARY ? SECONDARY : c);
        }
        return r;
    }
    if ((s == 0 || s == 0.5 || s == 1) && !has_primary && !has_secondary) {
        if (!has_any(VOWELS)) return ps;
        return restress(u32(1, SECONDARY) + ps);
    }
    if (s >= 1 && !has_primary && has_secondary) {
        u32 r = ps;
        for (auto& c : r) if (c == SECONDARY) c = PRIMARY;
        return r;
    }
    if (s > 1 && !has_primary && !has_secondary) {
        if (!has_any(VOWELS)) return ps;
        return restress(u32(1, PRIMARY) + ps);
    }
    return ps;
}

int stress_weight(const u32& ps) {
    int w = 0;
    for (char32_t c : ps) w += contains(DIPHTHONGS, c) ? 2 : 1;
    return w;
}

struct Tok {
    u32 text;
    std::string tag;
    u32 ws;
    std::optional<u32> ps;
    bool is_head = true;
    std::optional<u32> alias;
    Stress stress;
    std::optional<u32> currency;
    std::string num_flags;
    bool prespace = false;
    std::optional<int> rating;
};

struct Ctx {
    std::optional<bool> future_vowel;
    bool future_to = false;
};

using PR = std::pair<std::optional<u32>, std::optional<int>>;

Tok merge_tokens(const std::vector<Tok>& toks, const std::optional<u32>& unk) {
    Tok m;
    std::set<double> stresses;
    std::optional<u32> cur;
    bool rating_none = false;
    int rating_min = 1 << 30;
    for (const Tok& t : toks) {
        if (t.stress) stresses.insert(*t.stress);
        if (t.currency && (!cur || *t.currency > *cur)) cur = t.currency;
        if (!t.rating) rating_none = true;
        else rating_min = std::min(rating_min, *t.rating);
    }
    if (unk) {
        u32 ps;
        for (const Tok& t : toks) {
            if (t.prespace && !ps.empty() && !is_space(ps.back()) && t.ps && !t.ps->empty()) ps.push_back(' ');
            ps += t.ps ? *t.ps : *unk;
        }
        m.ps = ps;
    }
    for (size_t i = 0; i + 1 < toks.size(); ++i) m.text += toks[i].text + toks[i].ws;
    m.text += toks.back().text;
    int best = -1;
    for (const Tok& t : toks) {
        int w = 0;
        for (char32_t c : t.text) w += c == lower_c(c) ? 1 : 2;
        if (w > best) { best = w; m.tag = t.tag; }
    }
    m.ws = toks.back().ws;
    m.is_head = toks.front().is_head;
    if (stresses.size() == 1) m.stress = *stresses.begin();
    m.currency = cur;
    std::set<char> flags;
    for (const Tok& t : toks) for (char c : t.num_flags) flags.insert(c);
    for (char c : flags) m.num_flags.push_back(c);
    m.prespace = toks.front().prespace;
    if (!rating_none) m.rating = rating_min;
    return m;
}

// misaki's SUBTOKEN_REGEX, alternative by alternative in its order (findall semantics).
std::vector<u32> subtokenize(const u32& w) {
    std::vector<u32> out;
    const size_t n = w.size();
    auto L = [&](size_t i) { return i < n && is_letter(w[i]); };
    auto Q = [&](size_t i) { return i < n && contains(QUOTES, w[i]); };
    auto D = [&](size_t i) { return i < n && is_digit(w[i]); };
    auto S = [&](size_t i) { return i < n && (w[i] == ',' || w[i] == '.'); };
    size_t i = 0;
    while (i < n) {
        size_t len = 0;
        if (i == 0 && Q(0)) {                                                  // ^['‘’]+
            while (Q(len)) ++len;
        }
        if (!len && is_upper(w[i]) && i + 2 < n + 0 && i + 2 <= n - 1 && is_upper(w[i + 1]) && is_lower(w[i + 2])) {
            len = 1;                                                           // \p{Lu}(?=\p{Lu}\p{Ll})
        }
        if (!len) {                                                            // (?:^-)?(?:\d?[,.]?\d)+
            size_t j = i;
            if (i == 0 && w[0] == '-') j = 1;
            size_t k = j;
            for (;;) {
                if (D(k) && S(k + 1) && D(k + 2)) k += 3;
                else if (D(k) && D(k + 1)) k += 2;
                else if (S(k) && D(k + 1)) k += 2;
                else if (D(k)) k += 1;
                else break;
            }
            if (k > j) len = k - i;
        }
        if (!len && (w[i] == '-' || w[i] == '_')) {                            // [-_]+
            while (i + len < n && (w[i + len] == '-' || w[i + len] == '_')) ++len;
        }
        if (!len && Q(i) && Q(i + 1)) {                                        // ['‘’]{2,}
            while (Q(i + len)) ++len;
        }
        if (!len) {                                                            // \p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})
            for (size_t a = 0; !len && (a == 0 || L(i + a - 1)); ++a) {
                size_t pos = i + a;
                for (;;) {
                    if (pos + 1 < n && is_lower(w[pos]) && is_upper(w[pos + 1])) { len = pos + 1 - i; break; }
                    if (Q(pos) && L(pos + 1)) { pos += 2; continue; }
                    break;
                }
                if (!L(i + a)) break;
            }
        }
        if (!len && L(i)) {                                                    // \p{L}+(?:['‘’]\p{L})*
            size_t k = i;
            while (L(k)) ++k;
            while (Q(k) && L(k + 1)) { k += 2; while (L(k)) ++k; }
            len = k - i;
        }
        if (!len && !(w[i] == '-' || w[i] == '_' || is_letter(w[i]) || Q(i) || is_digit(w[i]))) len = 1;   // [^-_\p{L}'‘’\d]
        if (!len && Q(i)) {                                                    // ['‘’]+$
            size_t k = i;
            while (Q(k)) ++k;
            if (k == n) len = k - i;
        }
        if (!len) { ++i; continue; }
        out.push_back(w.substr(i, len));
        i += len;
    }
    return out;
}

// --- num2words, English ---------------------------------------------------------------
const char* kSmall[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                        "eighteen", "nineteen"};
const char* kTens[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};

std::string below_1000(int n) {
    std::string s;
    if (n >= 100) {
        s = std::string(kSmall[n / 100]) + " hundred";
        n %= 100;
        if (n) s += " and ";
        else return s;
    }
    if (n < 20) return s + kSmall[n];
    s += kTens[n / 10];
    if (n % 10) s += std::string("-") + kSmall[n % 10];
    return s;
}

}  // namespace

std::string num2words_cardinal(long long n) {
    if (n < 0) return "minus " + num2words_cardinal(-n);
    if (n < 1000) return below_1000((int)n);
    static const std::pair<long long, const char*> scales[] = {
        {1000000000000000000LL, "quintillion"}, {1000000000000000LL, "quadrillion"},
        {1000000000000LL, "trillion"}, {1000000000LL, "billion"}, {1000000LL, "million"}, {1000LL, "thousand"}};
    std::string s;
    long long rest = n;
    for (const auto& [v, name] : scales) {
        if (rest >= v) {
            if (!s.empty()) s += ", ";
            s += num2words_cardinal(rest / v) + " " + name;
            rest %= v;
        }
    }
    if (rest) s += (rest < 100 ? " and " : ", ") + below_1000((int)rest);
    return s;
}

std::string num2words_ordinal(long long n) {
    std::string c = num2words_cardinal(n);
    size_t sp = c.find_last_of(" -");
    std::string head = sp == std::string::npos ? "" : c.substr(0, sp + 1);
    std::string last = sp == std::string::npos ? c : c.substr(sp + 1);
    static const std::pair<const char*, const char*> ords[] = {
        {"one", "first"}, {"two", "second"}, {"three", "third"}, {"four", "fourth"}, {"five", "fifth"},
        {"six", "sixth"}, {"seven", "seventh"}, {"eight", "eighth"}, {"nine", "ninth"}, {"ten", "tenth"},
        {"eleven", "eleventh"}, {"twelve", "twelfth"}};
    for (const auto& [a, b] : ords)
        if (last == a) return head + b;
    if (!last.empty() && last.back() == 'y') last = last.substr(0, last.size() - 1) + "ie";
    return head + last + "th";
}

std::string num2words_year(long long v) {
    std::string suffix;
    if (v < 0) { v = -v; suffix = " BC"; }
    const long long high = v / 100, low = v % 100;
    std::string t;
    if (high == 0 || (high % 10 == 0 && low < 10) || high >= 100) t = num2words_cardinal(v);
    else {
        const std::string ht = num2words_cardinal(high);
        if (low == 0) t = ht + " hundred";
        else if (low < 10) t = ht + " oh-" + num2words_cardinal(low);
        else t = ht + " " + num2words_cardinal(low);
    }
    return t + suffix;
}

namespace {

// re.split(r'[^a-z]+', s): a run of separators splits once, so only the ends can be empty.
std::vector<std::string> split_words(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    bool in_sep = false;
    for (char c : s) {
        if (c >= 'a' && c <= 'z') {
            cur.push_back(c);
            in_sep = false;
        } else if (!in_sep) {
            out.push_back(cur);
            cur.clear();
            in_sep = true;
        }
    }
    out.push_back(cur);
    return out;
}

class Lexicon {
public:
    Lexicon(const std::string& gold, const std::string& silver, bool british) : british_(british) {
        load(gold, golds_);
        load(silver, silvers_);
    }
    bool british() const { return british_; }

    PR call(const Tok& tk, const Ctx& ctx) const {
        u32 word = tk.alias ? *tk.alias : tk.text;
        word = replace_all(replace_all(word, U("‘"), U("'")), U("’"), U("'"));
        Stress stress;
        if (word != lower(word)) stress = word == upper(word) ? 2.0 : 0.5;
        PR r = get_word(word, tk.tag, stress, ctx);
        if (r.first) return {apply_stress(append_currency(*r.first, tk.currency), tk.stress), r.second};
        if (is_number(word, tk.is_head)) {
            PR n;
            try {
                n = get_number(word, tk.currency, tk.is_head, tk.num_flags);
            } catch (const std::exception&) {   // a number past 64 bits reads as unknown, not as a crash
                return {std::nullopt, std::nullopt};
            }
            if (n.first) return {apply_stress(*n.first, tk.stress), n.second};
            return n;
        }
        return {std::nullopt, std::nullopt};
    }

    PR get_nnp(const u32& word) const {
        u32 ps;
        for (char32_t c : word) {
            if (!is_letter(c)) continue;
            const u32 letters = fold_latin(c);
            if (letters.empty()) return {std::nullopt, std::nullopt};
            for (char32_t l : letters) {
                auto it = golds_.find(u32(1, l));
                if (it == golds_.end() || it->second.table) return {std::nullopt, std::nullopt};
                ps += it->second.ps;
            }
        }
        ps = apply_stress(ps, 0.0);
        const size_t at = ps.rfind(SECONDARY);
        if (at != u32::npos) ps[at] = PRIMARY;
        return {ps, 3};
    }

    PR lookup(u32 word, std::string tag, Stress stress, const Ctx* ctx) const {
        std::optional<bool> is_nnp;
        if (word == upper(word) && !golds_.count(word)) {
            word = lower(word);
            is_nnp = tag == "NNP";
        }
        const LexEntry* e = find(golds_, word);
        int rating = 4;
        if (!e && !(is_nnp && *is_nnp)) {
            e = find(silvers_, word);
            rating = 3;
        }
        std::optional<u32> ps;
        if (e && !e->table) ps = e->ps;
        else if (e) {
            if (ctx && !ctx->future_vowel && has_tag(*e, "None")) tag = "None";
            else if (!has_tag(*e, tag)) tag = parent_tag(tag);
            ps = get_tag(*e, tag);
            if (!has_tag(*e, tag)) ps = get_tag(*e, "DEFAULT");
        }
        if (!ps || (is_nnp && *is_nnp && ps->find(PRIMARY) == u32::npos)) {
            PR n = get_nnp(word);
            if (n.first) return n;
        }
        if (!ps) return {std::nullopt, std::nullopt};
        return {apply_stress(*ps, stress), rating};
    }

    bool is_known(const u32& word, const std::string&) const {
        if (golds_.count(word) || symbol(word) || silvers_.count(word)) return true;
        if (!isalpha_str(word) || !lexicon_ords(word)) return false;
        if (word.size() == 1) return true;
        if (word == upper(word) && golds_.count(lower(word))) return true;
        return word.substr(1) == upper(word.substr(1));
    }

    u32 s_(const u32& stem) const {
        if (stem.empty()) return stem;
        if (contains(U("ptkfθ"), stem.back())) return stem + U("s");
        if (contains(U("szʃʒʧʤ"), stem.back())) return stem + (british_ ? U("ɪz") : U("ᵻz"));
        return stem + U("z");
    }
    PR stem_s(const u32& word, const std::string& tag, Stress stress, const Ctx* ctx) const {
        if (word.size() < 3 || !ends_with(word, "s")) return {};
        u32 stem;
        if (!ends_with(word, "ss") && is_known(word.substr(0, word.size() - 1), tag)) stem = word.substr(0, word.size() - 1);
        else if ((ends_with(word, "'s") || (word.size() > 4 && ends_with(word, "es") && !ends_with(word, "ies"))) &&
                 is_known(word.substr(0, word.size() - 2), tag)) stem = word.substr(0, word.size() - 2);
        else if (word.size() > 4 && ends_with(word, "ies") && is_known(word.substr(0, word.size() - 3) + U("y"), tag))
            stem = word.substr(0, word.size() - 3) + U("y");
        else return {};
        PR r = lookup(stem, tag, stress, ctx);
        if (!r.first) return {std::nullopt, r.second};
        return {s_(*r.first), r.second};
    }
    u32 ed_(const u32& stem) const {
        if (stem.empty()) return stem;
        const char32_t c = stem.back();
        if (contains(U("pkfθʃsʧ"), c)) return stem + U("t");
        if (c == 'd') return stem + (british_ ? U("ɪd") : U("ᵻd"));
        if (c != 't') return stem + U("d");
        if (british_ || stem.size() < 2) return stem + U("ɪd");
        if (contains(US_TAUS, stem[stem.size() - 2])) return stem.substr(0, stem.size() - 1) + U("ɾᵻd");
        return stem + U("ᵻd");
    }
    PR stem_ed(const u32& word, const std::string& tag, Stress stress, const Ctx* ctx) const {
        if (word.size() < 4 || !ends_with(word, "d")) return {};
        u32 stem;
        if (!ends_with(word, "dd") && is_known(word.substr(0, word.size() - 1), tag)) stem = word.substr(0, word.size() - 1);
        else if (word.size() > 4 && ends_with(word, "ed") && !ends_with(word, "eed") && is_known(word.substr(0, word.size() - 2), tag))
            stem = word.substr(0, word.size() - 2);
        else return {};
        PR r = lookup(stem, tag, stress, ctx);
        if (!r.first) return {std::nullopt, r.second};
        return {ed_(*r.first), r.second};
    }
    std::optional<u32> ing_(const u32& stem) const {
        if (stem.empty()) return stem;
        if (british_) {
            if (stem.back() == U("ə")[0] || stem.back() == U("ː")[0]) return std::nullopt;
        } else if (stem.size() > 1 && stem.back() == 't' && contains(US_TAUS, stem[stem.size() - 2])) {
            return stem.substr(0, stem.size() - 1) + U("ɾɪŋ");
        }
        return stem + U("ɪŋ");
    }
    PR stem_ing(const u32& word, const std::string& tag, Stress stress, const Ctx* ctx) const {
        if (word.size() < 5 || !ends_with(word, "ing")) return {};
        const u32 base = word.substr(0, word.size() - 3);
        u32 stem;
        if (word.size() > 5 && is_known(base, tag)) stem = base;
        else if (is_known(base + U("e"), tag)) stem = base + U("e");
        else if (word.size() > 5 && doubled_ing(word) && is_known(word.substr(0, word.size() - 4), tag))
            stem = word.substr(0, word.size() - 4);
        else return {};
        PR r = lookup(stem, tag, stress, ctx);
        if (!r.first) return {std::nullopt, r.second};
        std::optional<u32> p = ing_(*r.first);
        return {p, r.second};
    }

    PR get_word(u32 word, const std::string& tag, Stress stress, const Ctx& ctx) const {
        PR sc = special_case(word, tag, stress, ctx);
        if (sc.first) return sc;
        const u32 wl = lower(word);
        u32 noq;
        for (char32_t c : word) if (c != '\'') noq.push_back(c);
        if (word.size() > 1 && isalpha_str(noq) && word != wl && (tag != "NNP" || word.size() > 7) &&
            !golds_.count(word) && !silvers_.count(word) && (word == upper(word) || word.substr(1) == lower(word.substr(1))) &&
            (golds_.count(wl) || silvers_.count(wl) || stem_s(wl, tag, stress, &ctx).first ||
             stem_ed(wl, tag, stress, &ctx).first || stem_ing(wl, tag, stress, &ctx).first))
            word = wl;
        if (is_known(word, tag)) return lookup(word, tag, stress, &ctx);
        if (ends_with(word, "s'") && is_known(word.substr(0, word.size() - 2) + U("'s"), tag))
            return lookup(word.substr(0, word.size() - 2) + U("'s"), tag, stress, &ctx);
        if (ends_with(word, "'") && is_known(word.substr(0, word.size() - 1), tag))
            return lookup(word.substr(0, word.size() - 1), tag, stress, &ctx);
        PR r = stem_s(word, tag, stress, &ctx);
        if (r.first) return r;
        r = stem_ed(word, tag, stress, &ctx);
        if (r.first) return r;
        r = stem_ing(word, tag, stress ? stress : Stress(0.5), &ctx);
        if (r.first) return r;
        return {};
    }

private:
    static void load(const std::string& path, LexMap& m) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open lexicon " + path);
        const auto j = nlohmann::json::parse(in);
        LexMap raw;
        for (auto it = j.begin(); it != j.end(); ++it) {
            LexEntry e;
            if (it.value().is_string()) e.ps = utf8_to_u32(it.value().get<std::string>());
            else if (it.value().is_object()) {
                e.table = true;
                for (auto t = it.value().begin(); t != it.value().end(); ++t)
                    e.by_tag.push_back({t.key(), t.value().is_null() ? std::nullopt
                                                                     : std::optional<u32>(utf8_to_u32(t.value().get<std::string>()))});
            } else continue;
            raw[utf8_to_u32(it.key())] = std::move(e);
        }
        // grow_dictionary: lowercase <-> Capitalized aliases, the file's own keys win.
        for (const auto& [k, v] : raw) {
            if (k.size() < 2) continue;
            if (k == lower(k)) {
                if (k != capitalize(k)) m.emplace(capitalize(k), v);
            } else if (k == capitalize(lower(k))) {
                m.emplace(lower(k), v);
            }
        }
        for (auto& [k, v] : raw) m[k] = v;
    }
    static const LexEntry* find(const LexMap& m, const u32& w) {
        auto it = m.find(w);
        return it == m.end() ? nullptr : &it->second;
    }
    static bool has_tag(const LexEntry& e, const std::string& t) {
        for (auto& [k, v] : e.by_tag) if (k == t) return true;
        return false;
    }
    static std::optional<u32> get_tag(const LexEntry& e, const std::string& t) {
        for (auto& [k, v] : e.by_tag) if (k == t) return v;
        return std::nullopt;
    }
    static std::string parent_tag(const std::string& tag) {
        if (starts_with(tag, "VB")) return "VERB";
        if (starts_with(tag, "NN")) return "NOUN";
        if (starts_with(tag, "ADV") || starts_with(tag, "RB")) return "ADV";
        if (starts_with(tag, "ADJ") || starts_with(tag, "JJ")) return "ADJ";
        return tag;
    }
    static const char* symbol(const u32& w) {
        if (w == U("%")) return "percent";
        if (w == U("&")) return "and";
        if (w == U("+")) return "plus";
        if (w == U("@")) return "at";
        return nullptr;
    }
    static bool doubled_ing(const u32& w) {
        if (ends_with(w, "cking")) return true;
        if (w.size() < 5) return false;
        const char32_t a = w[w.size() - 5], b = w[w.size() - 4];
        return a == b && contains(U("bcdgklmnprstvxz"), a);
    }
    u32 gold_ps(const char* w) const {
        const LexEntry* e = find(golds_, U(w));
        if (!e) return U("");
        if (!e->table) return e->ps;
        auto d = get_tag(*e, "DEFAULT");
        return d ? *d : U("");
    }
    u32 gold_tag(const char* w, const char* t) const {
        const LexEntry* e = find(golds_, U(w));
        if (!e || !e->table) return gold_ps(w);
        auto d = get_tag(*e, t);
        return d ? *d : gold_ps(w);
    }

    PR special_case(const u32& word, const std::string& tag, Stress stress, const Ctx& ctx) const {
        if (tag == "ADD" && (word == U(".") || word == U("/")))
            return lookup(word == U(".") ? U("dot") : U("slash"), "", -0.5, &ctx);
        if (const char* s = symbol(word)) return lookup(U(s), "", std::nullopt, &ctx);
        {
            u32 stripped = word;
            while (!stripped.empty() && stripped.front() == '.') stripped.erase(0, 1);
            while (!stripped.empty() && stripped.back() == '.') stripped.pop_back();
            u32 nodot;
            size_t longest = 0, cur = 0;
            for (char32_t c : word) {
                if (c == '.') { longest = std::max(longest, cur); cur = 0; }
                else { nodot.push_back(c); ++cur; }
            }
            longest = std::max(longest, cur);
            if (stripped.find('.') != u32::npos && isalpha_str(nodot) && longest < 3) return get_nnp(word);
        }
        if (word == U("a") || word == U("A")) return {tag == "DT" ? U("ɐ") : U("ˈA"), 4};
        if (word == U("am") || word == U("Am") || word == U("AM")) {
            if (starts_with(tag, "NN")) return get_nnp(word);
            if (!ctx.future_vowel || word != U("am") || (stress && *stress > 0)) return {gold_ps("am"), 4};
            return {U("ɐm"), 4};
        }
        if (word == U("an") || word == U("An") || word == U("AN")) {
            if (word == U("AN") && starts_with(tag, "NN")) return get_nnp(word);
            return {U("ɐn"), 4};
        }
        if (word == U("I") && tag == "PRP") return {U("ˌI"), 4};
        if ((word == U("by") || word == U("By") || word == U("BY")) && parent_tag(tag) == "ADV") return {U("bˈI"), 4};
        if (word == U("to") || word == U("To") || (word == U("TO") && (tag == "TO" || tag == "IN"))) {
            if (!ctx.future_vowel) return {gold_ps("to"), 4};
            return {*ctx.future_vowel ? U("tʊ") : U("tə"), 4};
        }
        if (word == U("in") || word == U("In") || (word == U("IN") && tag != "NNP")) {
            const u32 s = (!ctx.future_vowel || tag != "IN") ? U("ˈ") : U("");
            return {s + U("ɪn"), 4};
        }
        if (word == U("the") || word == U("The") || (word == U("THE") && tag == "DT"))
            return {(ctx.future_vowel && *ctx.future_vowel) ? U("ði") : U("ðə"), 4};
        if (tag == "IN") {
            const u32 l = lower(word);
            if (l == U("vs") || l == U("vs.")) return lookup(U("versus"), "", std::nullopt, &ctx);
        }
        if (word == U("used") || word == U("Used") || word == U("USED")) {
            if ((tag == "VBD" || tag == "JJ") && ctx.future_to) return {gold_tag("used", "VBD"), 4};
            return {gold_tag("used", "DEFAULT"), 4};
        }
        return {};
    }

    static bool is_currency_amount(const u32& word) {
        const size_t d = word.find('.');
        if (d == u32::npos) return true;
        if (std::count(word.begin(), word.end(), U'.') > 1) return false;
        return word.size() - d - 1 < 3;
    }

    static bool is_number(u32 word, bool is_head) {
        bool any = false;
        for (char32_t c : word) if (is_digit(c)) any = true;
        if (!any) return false;
        for (const char* s : {"ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"})
            if (ends_with(word, s)) { word = word.substr(0, word.size() - U(s).size()); break; }
        for (size_t i = 0; i < word.size(); ++i) {
            const char32_t c = word[i];
            if (!(is_digit(c) || c == ',' || c == '.' || (is_head && i == 0 && c == '-'))) return false;
        }
        return true;
    }

    PR get_number(u32 word, const std::optional<u32>& currency, bool is_head, const std::string& flags) const {
        u32 suffix;
        {
            size_t k = word.size();
            while (k > 0 && ((word[k - 1] >= 'a' && word[k - 1] <= 'z') || word[k - 1] == '\'')) --k;
            suffix = word.substr(k);
            word = word.substr(0, k);
        }
        const bool ordinal_sfx = suffix == U("st") || suffix == U("nd") || suffix == U("rd") || suffix == U("th");
        std::vector<std::pair<u32, int>> result;
        auto push = [&](const PR& r) { if (r.first) result.push_back({*r.first, r.second ? *r.second : 3}); };
        if (!word.empty() && word[0] == '-') {
            push(lookup(U("minus"), "", std::nullopt, nullptr));
            word = word.substr(1);
        }
        const bool flag_amp = flags.find('&') != std::string::npos, flag_a = flags.find('a') != std::string::npos,
                   flag_n = flags.find('n') != std::string::npos;
        auto extend = [&](const std::string& words, bool first) {
            const std::vector<std::string> splits = split_words(words);
            for (size_t i = 0; i < splits.size(); ++i) {
                const std::string& w = splits[i];
                if (w != "and" || flag_amp) {
                    if (first && i == 0 && splits.size() > 1 && w == "one" && flag_a) result.push_back({U("ə"), 4});
                    else push(lookup(utf8_to_u32(w), "", w == "point" ? Stress(-2.0) : std::nullopt, nullptr));
                } else if (w == "and" && flag_n && !result.empty()) {
                    result.back().first += U("ən");
                }
            }
        };
        auto extend_num = [&](const u32& digits, bool first) { extend(num2words_cardinal(std::stoll(u32_to_utf8(digits))), first); };
        auto no_commas = [](const u32& s) { u32 r; for (char32_t c : s) if (c != ',') r.push_back(c); return r; };
        const bool cur_known = currency && (*currency == U("$") || *currency == U("£") || *currency == U("€"));
        if (word.empty()) return {};
        if (all_digits(word) && ordinal_sfx) {
            extend(num2words_ordinal(std::stoll(u32_to_utf8(word))), true);
        } else if (result.empty() && word.size() == 4 && !cur_known && all_digits(word)) {
            extend(num2words_year(std::stoll(u32_to_utf8(word))), true);
        } else if (!is_head && word.find('.') == u32::npos) {
            const u32 num = no_commas(word);
            if (num.empty()) return {};
            if (num[0] == '0' || num.size() > 3) {
                for (char32_t c : num) extend_num(u32(1, c), false);
            } else if (num.size() == 3 && !ends_with(num, "00")) {
                extend_num(num.substr(0, 1), true);
                if (num[1] == '0') {
                    push(lookup(U("O"), "", -2.0, nullptr));
                    extend_num(num.substr(2, 1), false);
                } else {
                    extend_num(num.substr(1), false);
                }
            } else {
                extend_num(num, true);
            }
        } else if (std::count(word.begin(), word.end(), U'.') > 1 || !is_head) {
            bool first = true;
            u32 part;
            const u32 w2 = no_commas(word) + U(".");
            for (char32_t c : w2) {
                if (c != '.') { part.push_back(c); continue; }
                if (!part.empty()) {
                    bool nonzero_tail = false;
                    for (size_t q = 1; q < part.size(); ++q) if (part[q] != '0') nonzero_tail = true;
                    if (part[0] == '0' || (part.size() != 2 && nonzero_tail)) {
                        for (char32_t d : part) extend_num(u32(1, d), false);
                    } else {
                        extend_num(part, first);
                    }
                }
                first = false;
                part.clear();
            }
        } else if (cur_known && is_currency_amount(word)) {
            static const std::pair<const char*, const char*> units[] = {{"dollar", "cent"}, {"pound", "pence"}, {"euro", "cent"}};
            const int ci = *currency == U("$") ? 0 : *currency == U("£") ? 1 : 2;
            const u32 w2 = no_commas(word);
            std::vector<std::pair<long long, std::string>> pairs;
            const size_t d = w2.find('.');
            const u32 a = d == u32::npos ? w2 : w2.substr(0, d);
            pairs.push_back({a.empty() ? 0 : std::stoll(u32_to_utf8(a)), units[ci].first});
            if (d != u32::npos) {
                const u32 b = w2.substr(d + 1);
                pairs.push_back({b.empty() ? 0 : std::stoll(u32_to_utf8(b)), units[ci].second});
            }
            if (pairs.size() > 1) {
                if (pairs[1].first == 0) pairs.resize(1);
                else if (pairs[0].first == 0) pairs.erase(pairs.begin());
            }
            for (size_t i = 0; i < pairs.size(); ++i) {
                if (i > 0) push(lookup(U("and"), "", std::nullopt, nullptr));
                extend(num2words_cardinal(pairs[i].first), i == 0);
                const std::string& unit = pairs[i].second;
                if (std::llabs(pairs[i].first) != 1 && unit != "pence") push(stem_s(utf8_to_u32(unit + "s"), "", std::nullopt, nullptr));
                else push(lookup(utf8_to_u32(unit), "", std::nullopt, nullptr));
            }
        } else {
            std::string words;
            if (all_digits(word)) {
                words = num2words_cardinal(std::stoll(u32_to_utf8(word)));
            } else if (word.find('.') == u32::npos) {
                const long long v = std::stoll(u32_to_utf8(no_commas(word)));
                words = ordinal_sfx ? num2words_ordinal(v) : num2words_cardinal(v);
            } else {
                const u32 w2 = no_commas(word);
                if (w2[0] == '.') {
                    words = "point";
                    for (size_t q = 1; q < w2.size(); ++q) words += " " + num2words_cardinal(w2[q] - '0');
                } else {
                    // num2words(float(word)): the fraction's trailing zeros vanish in float -> str
                    const size_t d = w2.find('.');
                    u32 frac = w2.substr(d + 1);
                    while (!frac.empty() && frac.back() == '0') frac.pop_back();
                    words = num2words_cardinal(std::stoll(u32_to_utf8(w2.substr(0, d))));
                    if (frac.empty()) words += " point zero";
                    else {
                        words += " point";
                        for (char32_t c : frac) words += " " + num2words_cardinal(c - '0');
                    }
                }
            }
            extend(words, true);
        }
        if (result.empty()) return {};
        u32 ps;
        int rating = 1 << 30;
        for (size_t i = 0; i < result.size(); ++i) {
            if (i) ps.push_back(' ');
            ps += result[i].first;
            rating = std::min(rating, result[i].second);
        }
        if (suffix == U("s") || suffix == U("'s")) return {s_(ps), rating};
        if (suffix == U("ed") || suffix == U("'d")) return {ed_(ps), rating};
        if (suffix == U("ing")) {
            auto p = ing_(ps);
            return {p, rating};
        }
        return {ps, rating};
    }

    u32 append_currency(const u32& ps, const std::optional<u32>& currency) const {
        if (!currency) return ps;
        const char* unit = *currency == U("$") ? "dollars" : *currency == U("£") ? "pounds" : *currency == U("€") ? "euros" : nullptr;
        if (!unit) return ps;
        PR s = stem_s(U(unit), "", std::nullopt, nullptr);
        return s.first ? ps + U(" ") + *s.first : ps;
    }

    bool british_;
    LexMap golds_, silvers_;
};

// spaCy stand-in: whitespace chunks, then the punctuation spaCy splits off as its own token.
std::vector<Tok> tokenize(const u32& text, const Lexicon& lex) {
    std::vector<Tok> out;
    auto punct_tag = [](char32_t c, bool prefix) -> std::string {
        switch (c) {
            case '"': return prefix ? "``" : "''";
            case U'“': return "``";
            case U'”': return "''";
            case '\'': case U'‘': case U'’': return prefix ? "``" : "''";
            case '(': case '[': case '{': return "-LRB-";
            case ')': case ']': case '}': return "-RRB-";
            case ',': return ",";
            case '.': case '!': case '?': return ".";
            case ';': case ':': case U'…': case U'—': case U'–': return ":";
            case '$': case U'£': case U'€': return "$";
            case '#': return "#";
            default: return "";
        }
    };
    const u32 prefix_chars = U("\"“‘'([{$£€#");
    const u32 suffix_chars = U("\"”’'.,!?;:)]}…");
    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        while (i < n && is_space(text[i])) ++i;
        if (i >= n) break;
        size_t j = i;
        while (j < n && !is_space(text[j])) ++j;
        u32 chunk = text.substr(i, j - i);
        const u32 ws = j < n ? U(" ") : U("");
        i = j;
        std::vector<Tok> pre, post;
        // em/en dashes and "--" inside a chunk are their own tokens
        std::vector<u32> parts;
        {
            u32 cur;
            for (size_t k = 0; k < chunk.size(); ++k) {
                const bool dd = chunk[k] == '-' && k + 1 < chunk.size() && chunk[k + 1] == '-';
                if (chunk[k] == U'—' || chunk[k] == U'–' || dd) {
                    if (!cur.empty()) parts.push_back(cur);
                    cur.clear();
                    parts.push_back(dd ? U("--") : u32(1, chunk[k]));
                    if (dd) ++k;
                } else {
                    cur.push_back(chunk[k]);
                }
            }
            if (!cur.empty()) parts.push_back(cur);
        }
        for (size_t p = 0; p < parts.size(); ++p) {
            u32 w = parts[p];
            const bool last = p + 1 == parts.size();
            if (w == U("--") || w == U("—") || w == U("–") || w == U("-")) {
                Tok t;
                t.text = w == U("--") ? U("-") : w;
                t.tag = ":";
                t.ws = last ? ws : U("");
                out.push_back(t);
                continue;
            }
            pre.clear();
            post.clear();
            while (!w.empty() && contains(prefix_chars, w.front()) && w.size() > 1) {
                Tok t;
                t.text = u32(1, w.front());
                t.tag = punct_tag(w.front(), true);
                pre.push_back(t);
                w.erase(0, 1);
            }
            for (;;) {
                if (w.size() <= 1 || !contains(suffix_chars, w.back())) break;
                const char32_t c = w.back();
                if (c == '.') {
                    // keep abbreviations whole, as spaCy's exceptions do: U.S., e.g., Dr., Mrs.
                    bool abbrev = lex.is_known(w, "") && w.size() <= 5;
                    if (!abbrev) {
                        bool pattern = w.size() >= 2;
                        for (size_t q = 0; q < w.size(); ++q)
                            if ((q % 2 == 0 && !is_letter(w[q])) || (q % 2 == 1 && w[q] != '.')) pattern = false;
                        abbrev = pattern && w.size() % 2 == 0;
                    }
                    if (abbrev) break;
                    if (w.size() >= 3 && w[w.size() - 2] == '.' && w[w.size() - 3] == '.') {
                        Tok t;
                        t.text = U("...");
                        t.tag = ":";
                        post.insert(post.begin(), t);
                        w.erase(w.size() - 3);
                        continue;
                    }
                }
                if ((c == '\'' || c == U'’') && w.size() > 1 && (w[w.size() - 2] == 's' || w[w.size() - 2] == 'S')) break;
                Tok t;
                t.text = u32(1, c);
                t.tag = punct_tag(c, false);
                post.insert(post.begin(), t);
                w.pop_back();
            }
            std::vector<Tok> seq = pre;
            if (!w.empty()) {
                Tok t;
                t.text = w;
                bool digit = false;
                for (char32_t c : w) if (is_digit(c)) digit = true;
                t.tag = digit ? "CD" : "";
                if (w.size() == 1 && punct_tag(w[0], false) != "" && !is_letter(w[0]) && !is_digit(w[0])) t.tag = punct_tag(w[0], true);
                seq.push_back(t);
            }
            seq.insert(seq.end(), post.begin(), post.end());
            for (size_t q = 0; q < seq.size(); ++q) {
                seq[q].ws = (last && q + 1 == seq.size()) ? ws : U("");
                out.push_back(seq[q]);
            }
        }
    }
    // Fixed tags where misaki asks spaCy (OPEN-TTS-G2P lists them).
    for (size_t k = 0; k < out.size(); ++k) {
        Tok& t = out[k];
        if (!t.tag.empty()) continue;
        if (t.text == U("a")) t.tag = "DT";
        else if (t.text == U("A")) {
            bool sentence_start = k == 0 || out[k - 1].tag == "." || out[k - 1].tag == "``";
            t.tag = sentence_start ? "DT" : "NNP";
        } else if (t.text == U("I")) t.tag = "PRP";
        else if (lower(t.text) == U("in") || lower(t.text) == U("vs") || lower(t.text) == U("vs.")) t.tag = "IN";
        else if (lower(t.text) == U("used") && k + 1 < out.size() && lower(out[k + 1].text) == U("to")) t.tag = "VBD";
        else if (lower(t.text) == U("that")) {
            // a subject or determiner after it makes "that" a conjunction or relative (unstressed)
            static const std::set<u32> clause = {U("i"), U("you"), U("he"), U("she"), U("it"), U("we"), U("they"),
                                                 U("the"), U("a"), U("an"), U("there"), U("this"), U("my"),
                                                 U("your"), U("his"), U("her"), U("our"), U("their")};
            t.tag = (k + 1 < out.size() && clause.count(lower(out[k + 1].text))) ? "IN" : "DT";
        }
    }
    return out;
}

struct Word {
    bool group = false;
    std::vector<Tok> toks;
};

std::vector<Word> retokenize(const std::vector<Tok>& tokens) {
    std::vector<Word> words;
    std::optional<u32> currency;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const Tok& token = tokens[i];
        std::vector<Tok> tks;
        if (!token.alias && !token.ps) {
            for (const u32& t : subtokenize(token.text)) {
                Tok s = token;
                s.text = t;
                s.ws = U("");
                s.is_head = true;
                s.prespace = false;
                s.ps.reset();
                s.alias.reset();
                tks.push_back(s);
            }
            if (tks.empty()) continue;
        } else {
            tks.push_back(token);
        }
        tks.back().ws = token.ws;
        for (size_t j = 0; j < tks.size(); ++j) {
            Tok& tk = tks[j];
            if (tk.alias || tk.ps) {
            } else if (tk.tag == "$" && (tk.text == U("$") || tk.text == U("£") || tk.text == U("€"))) {
                currency = tk.text;
                tk.ps = U("");
                tk.rating = 4;
            } else if (tk.tag == ":" && (tk.text == U("-") || tk.text == U("–"))) {
                tk.ps = U("—");
                tk.rating = 3;
            } else if (PUNCT_TAGS.count(tk.tag) && !std::all_of(tk.text.begin(), tk.text.end(), [](char32_t c) {
                           const char32_t l = lower_c(c);
                           return l >= 97 && l <= 122;
                       })) {
                if (tk.tag == "-LRB-") tk.ps = U("(");
                else if (tk.tag == "-RRB-") tk.ps = U(")");
                else if (tk.tag == "``") tk.ps = U("“");
                else if (tk.tag == "\"\"" || tk.tag == "''") tk.ps = U("”");
                else {
                    u32 p;
                    for (char32_t c : tk.text) if (contains(PUNCTS, c)) p.push_back(c);
                    tk.ps = p;
                }
                tk.rating = 4;
            } else if (currency) {
                if (tk.tag != "CD") currency.reset();
                else if (j + 1 == tks.size() && (i + 1 == tokens.size() || tokens[i + 1].tag != "CD")) tk.currency = currency;
            } else if (j > 0 && j + 1 < tks.size() && tk.text == U("2") && is_letter(tks[j - 1].text.back()) &&
                       is_letter(tks[j + 1].text.front())) {
                tk.alias = U("to");
            }
            if (tk.alias || tk.ps) {
                words.push_back({false, {tk}});
            } else if (!words.empty() && words.back().group && words.back().toks.back().ws.empty()) {
                tk.is_head = false;
                words.back().toks.push_back(tk);
            } else if (!tk.ws.empty()) {
                words.push_back({false, {tk}});
            } else {
                words.push_back({true, {tk}});
            }
        }
    }
    for (Word& w : words)
        if (w.group && w.toks.size() == 1) w.group = false;
    return words;
}

Ctx token_context(const Ctx& ctx, const std::optional<u32>& ps, const Tok& token) {
    Ctx c;
    c.future_vowel = ctx.future_vowel;
    if (ps && !ps->empty()) {
        for (char32_t ch : *ps) {
            if (contains(VOWELS, ch) || contains(CONSONANTS, ch) || contains(NON_QUOTE_PUNCTS, ch)) {
                if (contains(NON_QUOTE_PUNCTS, ch)) c.future_vowel.reset();
                else c.future_vowel = contains(VOWELS, ch);
                break;
            }
        }
    }
    c.future_to = token.text == U("to") || token.text == U("To") ||
                  (token.text == U("TO") && (token.tag == "TO" || token.tag == "IN"));
    return c;
}

void resolve_tokens(std::vector<Tok>& tokens) {
    u32 text;
    for (size_t i = 0; i + 1 < tokens.size(); ++i) text += tokens[i].text + tokens[i].ws;
    text += tokens.back().text;
    std::set<int> classes;
    for (char32_t c : text)
        if (!contains(SUBTOKEN_JUNKS, c)) classes.insert(is_letter(c) ? 0 : is_digit(c) ? 1 : 2);
    const bool prespace = text.find(' ') != u32::npos || text.find('/') != u32::npos || classes.size() > 1;
    for (size_t i = 0; i < tokens.size(); ++i) {
        Tok& tk = tokens[i];
        if (!tk.ps) {
            if (i == tokens.size() - 1 && tk.text.size() == 1 && contains(NON_QUOTE_PUNCTS, tk.text[0])) {
                tk.ps = tk.text;
                tk.rating = 3;
            } else if (std::all_of(tk.text.begin(), tk.text.end(), [](char32_t c) { return contains(SUBTOKEN_JUNKS, c); })) {
                tk.ps = U("");
                tk.rating = 3;
            }
        } else if (i > 0) {
            tk.prespace = prespace;
        }
    }
    if (prespace) return;
    struct Ix { bool primary; int weight; size_t i; };
    std::vector<Ix> idx;
    for (size_t i = 0; i < tokens.size(); ++i)
        if (tokens[i].ps && !tokens[i].ps->empty())
            idx.push_back({tokens[i].ps->find(PRIMARY) != u32::npos, stress_weight(*tokens[i].ps), i});
    if (idx.size() == 2 && tokens[idx[0].i].text.size() == 1) {
        tokens[idx[1].i].ps = apply_stress(*tokens[idx[1].i].ps, -0.5);
        return;
    }
    int primaries = 0;
    for (auto& x : idx) primaries += x.primary;
    if (idx.size() < 2 || primaries <= (int)(idx.size() + 1) / 2) return;
    std::sort(idx.begin(), idx.end(), [](const Ix& a, const Ix& b) {
        if (a.primary != b.primary) return a.primary < b.primary;
        if (a.weight != b.weight) return a.weight < b.weight;
        return a.i < b.i;
    });
    idx.resize(idx.size() / 2);
    for (auto& x : idx) tokens[x.i].ps = apply_stress(*tokens[x.i].ps, -0.5);
}

}  // namespace

struct EnglishG2P::Impl {
    Lexicon lex;
    bool spell_unknown;
    Impl(const std::string& g, const std::string& s, bool b, bool spell) : lex(g, s, b), spell_unknown(spell) {}
};

EnglishG2P::EnglishG2P(const std::string& gold, const std::string& silver, bool british, bool spell_unknown)
    : impl_(std::make_unique<Impl>(gold, silver, british, spell_unknown)) {}
EnglishG2P::~EnglishG2P() = default;
bool EnglishG2P::british() const { return impl_->lex.british(); }

G2PResult EnglishG2P::operator()(const std::string& utf8) const {
    const Lexicon& lex = impl_->lex;
    u32 text = utf8_to_u32(utf8);
    for (auto& c : text)
        if (c == 0xA0 || c == '\t' || c == '\r' || c == '\n') c = ' ';
    std::vector<Word> words = retokenize(tokenize(text, lex));
    G2PResult res;
    std::vector<std::string> unknown_rev;
    Ctx ctx;
    for (size_t wi = words.size(); wi-- > 0;) {
        Word& w = words[wi];
        if (!w.group) {
            Tok& t = w.toks[0];
            if (!t.ps) {
                PR r = lex.call(t, ctx);
                t.ps = r.first;
                t.rating = r.second;
            }
            ctx = token_context(ctx, t.ps, t);
            continue;
        }
        std::vector<Tok>& g = w.toks;
        size_t left = 0, right = g.size();
        while (left < right) {
            bool blocked = false;
            for (size_t k = left; k < right; ++k) if (g[k].alias || g[k].ps) blocked = true;
            PR r;
            Tok merged;
            if (!blocked) {
                merged = merge_tokens(std::vector<Tok>(g.begin() + (long)left, g.begin() + (long)right), std::nullopt);
                r = lex.call(merged, ctx);
            }
            if (r.first) {
                g[left].ps = r.first;
                g[left].rating = r.second;
                for (size_t k = left + 1; k < right; ++k) g[k].ps = U("");
                ctx = token_context(ctx, r.first, merged);
                right = left;
                left = 0;
            } else if (left + 1 < right) {
                ++left;
            } else {
                --right;
                Tok& tk = g[right];
                if (!tk.ps && std::all_of(tk.text.begin(), tk.text.end(), [](char32_t c) { return contains(SUBTOKEN_JUNKS, c); })) {
                    tk.ps = U("");
                    tk.rating = 3;
                }
                left = 0;
            }
        }
        resolve_tokens(g);
    }
    // Unknown words: spelled letter by letter (default) or dropped, as KPipeline drops them without espeak.
    for (Word& w : words)
        for (Tok& t : w.toks)
            if (!t.ps) {
                res.unknown.push_back(u32_to_utf8(t.text));
                if (impl_->spell_unknown) {
                    PR r = lex.get_nnp(t.text);
                    if (r.first) t.ps = r.first;
                }
            }
    u32 out;
    for (Word& w : words) {
        Tok t = w.group ? merge_tokens(w.toks, U("")) : w.toks[0];
        u32 ps = t.ps ? *t.ps : U("");
        ps = replace_all(replace_all(ps, U("ɾ"), U("T")), U("ʔ"), U("t"));
        out += ps + t.ws;
    }
    while (!out.empty() && is_space(out.back())) out.pop_back();
    size_t lead = 0;
    while (lead < out.size() && is_space(out[lead])) ++lead;
    res.phonemes = u32_to_utf8(out.substr(lead));
    return res;
}

}  // namespace open_tts
