import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim =dim
        self.weight=weight
        self.eps=eps

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype= x.dtype
        x=x.astype(mx.float32)
        mean_square =mx.mean(mx.square(x),axis=-1,keepdims=True) +self.eps
        
        inv_rms=mx.rsqrt(mean_square)

        x_normed=x* inv_rms

        return (x_normed *self.weight).astype(orig_dtype)
        

        

