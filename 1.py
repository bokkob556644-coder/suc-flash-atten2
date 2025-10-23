import torch
import torch.nn.functional as F

def simple_flash_attention_example(batch_size=2, seq_len=16, embed_dim=64, num_heads=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("PyTorch version:", torch.__version__)
    print("CUDA version used by PyTorch:", torch.version.cuda)
    # طباعة ما إذا كان flash-SDP ممكن ومفعل
    print("flash_sdp_enabled (before):", torch.backends.cuda.flash_sdp_enabled())
    print("mem_efficient_sdp_enabled (before):", torch.backends.cuda.mem_efficient_sdp_enabled())
    print("math_sdp_enabled (before):", torch.backends.cuda.math_sdp_enabled())

    head_dim = embed_dim // num_heads

    Q = torch.randn(batch_size, seq_len, embed_dim,
                    device=device, dtype=torch.float16, requires_grad=True)
    K = torch.randn(batch_size, seq_len, embed_dim,
                    device=device, dtype=torch.float16, requires_grad=True)
    V = torch.randn(batch_size, seq_len, embed_dim,
                    device=device, dtype=torch.float16, requires_grad=True)

    Q = Q.view(batch_size, seq_len, num_heads, head_dim).permute(0,2,1,3)
    K = K.view(batch_size, seq_len, num_heads, head_dim).permute(0,2,1,3)
    V = V.view(batch_size, seq_len, num_heads, head_dim).permute(0,2,1,3)

    # نستخدم context manager لفرض استخدام FlashAttention إذا كان متوفّراً
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        print("Inside context — flash_sdp_enabled:", torch.backends.cuda.flash_sdp_enabled())
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )

    print("Output shape (before reshape):", out.shape)

    out = out.permute(0,2,1,3).contiguous().view(batch_size, seq_len, embed_dim)
    print("Final output shape:", out.shape)

    # اختبار التدرّج
    out_sum = out.sum()
    out_sum.backward()
    # لاحظ: Q ليس leaf tensor بعد view+permute لذا grad قد تكون None
    print("Gradient check: Q.grad is not None?", Q.grad is not None)

if __name__ == "__main__":
    simple_flash_attention_example()

