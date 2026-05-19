import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        logprobs = copy.copy(logprobs)  # TODO: do we really need a copy?
        if top_k is not None and top_k > 0:
            mask_elements = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[
                :, top_k:
            ]
            logprobs[:, mask_elements] = -mx.inf
        if top_p is not None and top_p > 0:
            sorted_idx = mx.argsort(-logprobs, axis=-1)
            sorted_logprobs = logprobs[:, sorted_idx]
            cumsum = mx.cumsum(mx.exp(sorted_logprobs), axis=-1)
            mask_elements = cumsum < top_p
            mask_elements[..., 0] = True
            logprobs[:, sorted_idx] = mx.where(mask_elements, sorted_logprobs, -mx.inf)
        logprobs = logprobs / temp
        return mx.random.categorical(logprobs, axis=-1)

    return sample


export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
export ANTHROPIC_AUTH_TOKEN=sk-8813e4494efc44b983a7b3a79f098854
export ANTHROPIC_MODEL=deepseek-v4-pro[1m]
export ANTHROPIC_DEFAULT_OPUS_MODEL=deepseek-v4-pro[1m]
export ANTHROPIC_DEFAULT_SONNET_MODEL=deepseek-v4-pro[1m]
export ANTHROPIC_DEFAULT_HAIKU_MODEL=deepseek-v4-flash
export CLAUDE_CODE_SUBAGENT_MODEL=deepseek-v4-flash
export CLAUDE_CODE_EFFORT_LEVEL=max

