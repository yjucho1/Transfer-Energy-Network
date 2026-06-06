# Lead-Selective Hybrid Sweep Summary

| Dataset | Pred | Best Config | MSE Delta | MAE Delta | CRPS Delta | Mean Q Delta | All Better |
|---|---:|---|---:|---:|---:|---:|---|
| `etth1` | `96` | `attn`, `scale=var_blend`, `target=crps`, `temp=1.0`, `usage=0.005`, `aux=0.0` | -0.003730 | -0.002865 | +0.000177 | -0.000187 | no |
| `etth1` | `192` | `attn`, `scale=var_blend`, `target=crps`, `temp=1.0`, `usage=0.005`, `aux=0.0` | -0.005664 | +0.001199 | -0.003136 | -0.001658 | no |
| `etth1` | `336` | `attn`, `scale=var_blend`, `target=crps`, `temp=1.0`, `usage=0.005`, `aux=0.0` | +0.002110 | +0.001266 | -0.001232 | -0.000583 | no |
| `etth1` | `720` | `attn`, `scale=var_blend`, `target=crps`, `temp=1.0`, `usage=0.005`, `aux=0.0` | -0.004447 | -0.001085 | -0.001107 | -0.000398 | yes |
