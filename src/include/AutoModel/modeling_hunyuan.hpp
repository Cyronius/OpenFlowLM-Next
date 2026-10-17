/// \file modeling_hunyuan.hpp
/// \brief Tencent HunYuan V1 dense family (Hy-MT2 translation models).
/// \note Open kernels only: no closed HunYuan engine exists, so a missing kernel set is a named refusal.

#pragma once
#include "AutoModel/automodel.hpp"
#ifdef OFLM_USE_OPEN_QWEN36
#include "open_qwen36/engine.hpp"
#endif

class HunYuan : public AutoModel {
private:
    void setup_tokenizer(std::string model_path);

public:
    HunYuan(oflm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object()) override;
};
