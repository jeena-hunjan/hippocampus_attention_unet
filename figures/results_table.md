| Run | Config change | Val mean Dice | Test anterior | Test posterior | Test mean |
|---|---|---|---|---|---|
| run1_baseline | baseline | 0.887 | 0.896 ± 0.039 | 0.882 ± 0.040 | 0.889 ± 0.037 |
| run2_affine | affine | 0.890 | 0.894 ± 0.039 | 0.880 ± 0.043 | 0.887 ± 0.039 |
| run3_elastic | elastic | 0.890 | 0.900 ± 0.039 | 0.877 ± 0.040 | 0.889 ± 0.037 |
| run4_affine_elastic | affine, elastic | 0.892 | 0.900 ± 0.039 | 0.881 ± 0.039 | 0.891 ± 0.037 |
| run5_lr_1e-4 | affine, elastic, lr 1e-4 | 0.890 | 0.900 ± 0.039 | 0.881 ± 0.039 | 0.890 ± 0.036 |
| run6_lr_3e-3 | affine, elastic, lr 3e-3 | 0.891 | 0.898 ± 0.036 | 0.880 ± 0.038 | 0.889 ± 0.035 |
| run7_weighted_ce | affine, elastic, weighted CE | 0.819 | 0.844 ± 0.037 | 0.808 ± 0.033 | 0.826 ± 0.033 |
| run8_wide | affine, elastic, channels 32-256 | 0.894 | 0.896 ± 0.038 | 0.882 ± 0.041 | 0.889 ± 0.037 |

Test Dice is mean ± standard deviation across 39 test cases. Final model (highest validation Dice): **run8_wide**.
