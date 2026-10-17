// Mono float PCM to flac, aac or opus through the bundled FFmpeg's built-in encoders (2026-10-17).
#include "audio/audio_encoder.hpp"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/log.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
}

namespace audio_encode {
namespace {

struct Format {
    const char* muxer;
    AVCodecID codec;
    int bit_rate;
};

bool lookup(const std::string& f, Format& out) {
    if (f == "flac") out = {"flac", AV_CODEC_ID_FLAC, 0};
    else if (f == "aac") out = {"adts", AV_CODEC_ID_AAC, 64000};
    else if (f == "opus") out = {"ogg", AV_CODEC_ID_OPUS, 64000};
    else return false;
    return true;
}

void check(int r, const std::string& what) {
    if (r >= 0) return;
    char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(r, buf, sizeof buf);
    throw std::runtime_error(what + ": " + buf);
}

int append(void* opaque, const uint8_t* buf, int n) {
    static_cast<std::string*>(opaque)->append(reinterpret_cast<const char*>(buf), (size_t)n);
    return n;
}

// The rate nearest above `want` that the encoder takes, or its highest.
int pick_rate(const AVCodecContext* c, const AVCodec* codec, int want) {
    const void* cfg = nullptr;
    int n = 0;
    if (avcodec_get_supported_config(c, codec, AV_CODEC_CONFIG_SAMPLE_RATE, 0, &cfg, &n) < 0 || !cfg || n == 0)
        return want;
    const int* r = static_cast<const int*>(cfg);
    int best = 0, top = 0;
    for (int i = 0; i < n; ++i) {
        if (r[i] == want) return want;
        top = std::max(top, r[i]);
        if (r[i] > want && (best == 0 || r[i] < best)) best = r[i];
    }
    return best ? best : top;
}

AVSampleFormat pick_format(const AVCodecContext* c, const AVCodec* codec) {
    const void* cfg = nullptr;
    int n = 0;
    if (avcodec_get_supported_config(c, codec, AV_CODEC_CONFIG_SAMPLE_FORMAT, 0, &cfg, &n) < 0 || !cfg || n == 0)
        return AV_SAMPLE_FMT_FLTP;
    const AVSampleFormat* f = static_cast<const AVSampleFormat*>(cfg);
    for (int i = 0; i < n; ++i)
        if (f[i] == AV_SAMPLE_FMT_S16 || f[i] == AV_SAMPLE_FMT_S16P) return f[i];   // flac: the WAV's 16 bits
    return f[0];
}

struct Session {
    AVCodecContext* c = nullptr;
    AVFormatContext* oc = nullptr;
    AVIOContext* io = nullptr;
    SwrContext* swr = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* pkt = nullptr;
    ~Session() {
        av_packet_free(&pkt);
        av_frame_free(&frame);
        swr_free(&swr);
        if (io) av_freep(&io->buffer);
        avio_context_free(&io);
        avformat_free_context(oc);
        avcodec_free_context(&c);
    }
};

void drain(Session& s, AVStream* st) {
    for (;;) {
        const int r = avcodec_receive_packet(s.c, s.pkt);
        if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) return;
        check(r, "encode");
        av_packet_rescale_ts(s.pkt, s.c->time_base, st->time_base);
        s.pkt->stream_index = st->index;
        check(av_interleaved_write_frame(s.oc, s.pkt), "mux");
    }
}

}  // namespace

bool supported(const std::string& format) {
    Format f;
    return lookup(format, f) && avcodec_find_encoder(f.codec) != nullptr;
}

const char* content_type(const std::string& format) {
    if (format == "flac") return "audio/flac";
    if (format == "aac") return "audio/aac";
    if (format == "opus") return "audio/ogg";
    return "application/octet-stream";
}

std::string encode_mono(const std::vector<float>& samples, int sample_rate, const std::string& format) {
    static std::once_flag quiet;
    std::call_once(quiet, [] { av_log_set_level(AV_LOG_ERROR); });   // the flac muxer warns it cannot seek back
    Format f;
    if (!lookup(format, f)) throw std::runtime_error("no encoder for " + format);
    const AVCodec* codec = avcodec_find_encoder(f.codec);
    if (!codec) throw std::runtime_error("this FFmpeg build has no " + format + " encoder");

    Session s;
    s.c = avcodec_alloc_context3(codec);
    if (!s.c) throw std::runtime_error("avcodec_alloc_context3");
    s.c->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;   // FFmpeg marks its own opus encoder experimental
    av_channel_layout_default(&s.c->ch_layout, 1);
    s.c->sample_rate = pick_rate(s.c, codec, sample_rate);
    s.c->sample_fmt = pick_format(s.c, codec);
    s.c->time_base = AVRational{1, s.c->sample_rate};
    if (f.bit_rate) s.c->bit_rate = f.bit_rate;

    check(avformat_alloc_output_context2(&s.oc, nullptr, f.muxer, nullptr), "muxer " + std::string(f.muxer));
    if (s.oc->oformat->flags & AVFMT_GLOBALHEADER) s.c->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    check(avcodec_open2(s.c, codec, nullptr), "open the " + format + " encoder");
    AVStream* st = avformat_new_stream(s.oc, nullptr);
    if (!st) throw std::runtime_error("avformat_new_stream");
    check(avcodec_parameters_from_context(st->codecpar, s.c), "stream parameters");
    st->time_base = s.c->time_base;

    std::string out;
    constexpr int kIoBuf = 1 << 16;
    auto* iobuf = static_cast<unsigned char*>(av_malloc(kIoBuf));
    if (!iobuf) throw std::runtime_error("av_malloc");
    s.io = avio_alloc_context(iobuf, kIoBuf, 1, &out, nullptr, append, nullptr);
    if (!s.io) {
        av_free(iobuf);
        throw std::runtime_error("avio_alloc_context");
    }
    s.oc->pb = s.io;
    s.oc->flags |= AVFMT_FLAG_CUSTOM_IO;
    check(avformat_write_header(s.oc, nullptr), "write the " + format + " header");

    // Resample and convert the whole utterance at once; it is seconds of mono audio.
    AVChannelLayout mono;
    av_channel_layout_default(&mono, 1);
    check(swr_alloc_set_opts2(&s.swr, &s.c->ch_layout, s.c->sample_fmt, s.c->sample_rate, &mono, AV_SAMPLE_FMT_FLT,
                              sample_rate, 0, nullptr), "resampler");
    check(swr_init(s.swr), "resampler init");
    const int bps = av_get_bytes_per_sample(s.c->sample_fmt);
    const int in_n = (int)samples.size();
    const int cap = swr_get_out_samples(s.swr, in_n) + 64;
    std::vector<uint8_t> pcm((size_t)std::max(cap, 1) * bps);
    uint8_t* dst = pcm.data();
    const uint8_t* src = reinterpret_cast<const uint8_t*>(samples.data());
    int got = swr_convert(s.swr, &dst, cap, in_n ? &src : nullptr, in_n);
    check(got, "resample");
    uint8_t* tail = pcm.data() + (size_t)got * bps;
    const int more = swr_convert(s.swr, &tail, cap - got, nullptr, 0);
    check(more, "resample flush");
    const int total = got + more;

    const bool variable = (codec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE) || s.c->frame_size <= 0;
    const int fs = variable ? 4096 : s.c->frame_size;
    const bool small_last = variable || (codec->capabilities & AV_CODEC_CAP_SMALL_LAST_FRAME);
    s.frame = av_frame_alloc();
    s.pkt = av_packet_alloc();
    if (!s.frame || !s.pkt) throw std::runtime_error("av_frame_alloc");
    for (int off = 0; off < total; off += fs) {
        const int n = std::min(fs, total - off);
        av_frame_unref(s.frame);
        s.frame->nb_samples = small_last ? n : fs;   // a fixed-size encoder gets its last frame padded with silence
        s.frame->format = s.c->sample_fmt;
        s.frame->sample_rate = s.c->sample_rate;
        check(av_channel_layout_copy(&s.frame->ch_layout, &s.c->ch_layout), "frame layout");
        check(av_frame_get_buffer(s.frame, 0), "frame buffer");
        std::memset(s.frame->data[0], 0, (size_t)s.frame->nb_samples * bps);
        std::memcpy(s.frame->data[0], pcm.data() + (size_t)off * bps, (size_t)n * bps);
        s.frame->pts = off;
        check(avcodec_send_frame(s.c, s.frame), "encode");
        drain(s, st);
    }
    check(avcodec_send_frame(s.c, nullptr), "encoder flush");
    drain(s, st);
    check(av_write_trailer(s.oc), "write the " + format + " trailer");
    avio_flush(s.io);
    return out;
}

}  // namespace audio_encode
