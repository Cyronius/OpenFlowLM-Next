#include "attn.h"
extern "C" {
#if ATTN_QLOCAL
// this core keeps only its own heads' q, so it needs its head offset
void attn_q(const float *__restrict qh ATTN_BIAS_PARM, const bfloat16 *__restrict qn,
            const float *__restrict cs, ATTN_QT *__restrict qs, int h, int h0) {
  attn_q_impl(qh ATTN_BIAS_ARG, qn, cs, qs, h, h0);
}
#else
void attn_q(const float *__restrict qh ATTN_BIAS_PARM, const bfloat16 *__restrict qn,
            const float *__restrict cs, ATTN_QT *__restrict qs, int h) {
  attn_q_impl(qh ATTN_BIAS_ARG, qn, cs, qs, h);
}
#endif
}
