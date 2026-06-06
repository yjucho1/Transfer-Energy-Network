# Lead-Selective Hybrid Sweep Summary

| Dataset | Pred | Best Config | MSE Delta | MAE Delta | CRPS Delta | Mean Q Delta | All Better |
|---|---:|---|---:|---:|---:|---:|---|
| `etth1` | `192` | `attn`, `temp=0.7`, `usage=0.005`, `aux=0.0` | +0.023271 | +0.022555 | +0.006542 | +0.003271 | no |
| `etth2` | `192` | `basic`, `temp=1.0`, `usage=0.005`, `aux=0.0` | -0.000046 | -0.000073 | -0.000131 | -0.000085 | yes |
| `etth1` | `336` | `basic`, `temp=1.0`, `usage=0.005`, `aux=0.0` | +0.006331 | +0.004438 | +0.001963 | +0.000833 | no |
| `etth2` | `336` | `basic`, `temp=0.7`, `usage=0.005`, `aux=0.0` | -0.005147 | -0.003599 | -0.002795 | -0.001300 | yes |
| `etth1` | `720` | `attn`, `temp=1.0`, `usage=0.005`, `aux=0.0` | -0.009276 | -0.002420 | -0.002430 | -0.001044 | yes |
| `etth2` | `720` | `attn`, `temp=1.0`, `usage=0.001`, `aux=0.0` | +0.005974 | +0.004086 | +0.002616 | +0.001108 | no |
