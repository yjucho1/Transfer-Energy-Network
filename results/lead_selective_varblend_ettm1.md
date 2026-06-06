# Lead-Selective Hybrid Sweep Summary

| Dataset | Pred | Best Config | MSE Delta | MAE Delta | CRPS Delta | Mean Q Delta | All Better |
|---|---:|---|---:|---:|---:|---:|---|
| `ettm1` | `96` | `attn`, `scale=var_blend`, `target=crps`, `temp=1.0`, `usage=0.001`, `aux=0.0` | -0.001519 | +0.000134 | -0.005623 | -0.001871 | no |
| `ettm1` | `192` | `attn`, `scale=var_blend`, `target=crps`, `temp=1.0`, `usage=0.001`, `aux=0.0` | -0.001889 | -0.001239 | -0.002277 | -0.001072 | yes |
| `ettm1` | `336` | `attn`, `scale=var_blend`, `target=crps`, `temp=1.0`, `usage=0.001`, `aux=0.0` | -0.005341 | -0.003380 | -0.002868 | -0.001267 | yes |
| `ettm1` | `720` | `attn`, `scale=var_blend`, `target=crps`, `temp=1.0`, `usage=0.001`, `aux=0.0` | -0.076989 | -0.042587 | -0.024788 | -0.012086 | yes |
