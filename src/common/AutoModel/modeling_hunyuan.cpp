/// \file modeling_hunyuan.cpp
/// \brief Tencent HunYuan V1 dense family. See modeling_hunyuan.hpp.

#include "AutoModel/modeling_hunyuan.hpp"

/************              HunYuan family            **************/
HunYuan::HunYuan(oflm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "HunYuan") {}

void HunYuan::load_model(std::string model_path, json model_info, int default_context_length,
                         bool enable_preemption) {
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);

#ifdef OFLM_USE_OPEN_QWEN36
    const std::string kernels = open_qwen36::Engine::find_kernels(*this->lm_config);
    const char* sel = std::getenv("OFLM_HUNYUAN_ENGINE");
    if (sel && std::string(sel) == "closed")
        throw std::runtime_error("OFLM_HUNYUAN_ENGINE=closed: not implemented -- this build has no closed "
                                 "HunYuan engine; build the open kernels instead");
    if (kernels.empty())
        throw std::runtime_error("no open kernels were found for " + this->lm_config->model_name +
                                 ". HunYuan runs on the open kernels only: build them with "
                                 "open_kernels/export_qwen36_kernels.py --model-dir <model dir>, or point "
                                 "OFLM_OPEN_KERNELS_DIR at a built set.");
    header_print("OFLM", "HunYuan on the open kernels (" + kernels + ")");
    auto eng = std::make_unique<open_qwen36::Engine>(*this->lm_config, this->npu_device_inst, this->MAX_L);
    eng->load_open_weights();
    this->lm_engine = std::move(eng);
#else
    throw std::runtime_error("HunYuan needs the open engine, which this build does not have "
                             "(OFLM_USE_OPEN_QWEN36 is off -- it requires the XRT backend, not HRX)");
#endif

    this->lm_engine->clear_context();
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    // Hy-MT2's model card values for the 1.8B and 7B (its generation_config.json says top_p 0.8)
    sampler_config config;
    config.top_k = 20;
    config.top_p = 0.6;
    config.min_p = 0.0;
    config.temperature = 0.7;
    config.rep_penalty = 1.05;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}

void HunYuan::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string HunYuan::apply_chat_template(nlohmann::ordered_json& messages,
                                         nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    if (!tools.empty())
        inputs.tools = tools;
    return this->_shared_apply_template(inputs);
}

bool HunYuan::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input,
                     std::function<bool()> is_cancelled) {
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    if (!input.messages.empty()) {
        templated_text = this->apply_chat_template(input.messages, input.tools);
    }
    else if (!input.prompt.empty()) {
        nlohmann::ordered_json messages;
        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        templated_text = this->apply_chat_template(messages);
    }

    std::vector<int> tokens = this->tokenizer->encode(templated_text);
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    return this->_shared_insert(meta_info, tokens, is_cancelled);
}

std::string HunYuan::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os,
                              std::function<bool()> is_cancelled) {
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

std::string HunYuan::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input,
                                          int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    if (!this->insert(meta_info, input, is_cancelled)) {
        return "";
    }
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}
