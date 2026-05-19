import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    if scale is None:
        d_k= query.shape[-1]
        scale=mx.rsqrt(d_k)
    
    scores = mx.matmul(query,key.swapaxes(-2,-1)) *scale

    if mask is not None:
        socres += mask

    attn_weight = mx.softmax(scores,axis=-1)

    attention =mx.matmul(attn_weight,value)

    return attention


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    if scale is None:
        d_k= query.shape[-1]
        scale=d_k ** -0.5
    
    expected_shape = query.shape

    H_q,L, D=query.shape[-3:]
    H, S, _ = key.shape[-3:]
    
    q_counter= H_q//H

    B = query.shape[:-3]


    query=query.reshape(*B,-1,H,q_counter,L,D)
    key=key.reshape(*B,-1,H,1,S,D)
    value=value.reshape(*B,-1,H,1,S,D)

    score =mx.matmul(query,key.swapaxes(-2,-1)) *scale

    if mask is not None:
        if mask == "causal":
            mask = causal_mask(L,S,score.dtype)
            score = score +mask
        else:
            #mask = mx.broadcast_to(mask,(*B,H_q,L,S))
            mask = mask.reshape(*B,-1,H,q_counter,L,S)
            score =score +mask

    attn_weight = mx.softmax(score,axis=-1)

    attention =mx.matmul(attn_weight,value)

    return attention.reshape(expected_shape)




class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        assert hidden_size % num_heads ==0
        self.head_dim=hidden_size // num_heads
        self.scale =mx.rsqrt(self.head_dim)
        assert wq.shape == (num_heads*self.head_dim,hidden_size)
        assert wk.shape == (num_heads*self.head_dim,hidden_size)
        assert wv.shape == (num_heads*self.head_dim,hidden_size)
        assert wo.shape == (hidden_size,num_heads*self.head_dim)
        self.wq=wq
        self.wk=wk
        self.wv=wv
        self.wo=wo
        

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        assert query.shape == key.shape == value.shape
        q= linear(query,self.wq)
        q= q.reshape(*q.shape[:-1],self.num_heads,self.head_dim).swapaxes(-3,-2)
        k= linear(key,self.wk)
        k= k.reshape(*k.shape[:-1],self.num_heads,self.head_dim).swapaxes(-3,-2)
        v= linear(value,self.wv)
        v= v.reshape(*v.shape[:-1],self.num_heads,self.head_dim).swapaxes(-3,-2)

        attention=scaled_dot_product_attention_simple(q,k,v,scale=self.scale,mask=mask).swapaxes(-3,-2)
        attention=attention.reshape(*attention.shape[:-2],self.hidden_size)

        return linear(attention,self.wo)

        






def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    mask = mx.tril(mx.ones((L,S)),k=S-L)
    mask = mx.where(mask,mx.array(0),mx.array(-mx.inf)).astype(dtype)
    return mask











def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
