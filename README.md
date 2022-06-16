# Outline2Story

This repository contains source code for paper [Outline to Story: Fine-grained Controllable Story Generation from Cascaded Events](https://arxiv.org/abs/2101.00822):

```
@article{fang2021outline,
  title={Outline to Story: Fine-grained Controllable Story Generation from Cascaded Events},
  author={Fang, Le and Zeng, Tao and Liu, Chaochun and Bo, Liefeng and Dong, Wen and Chen, Changyou},
  journal={arXiv preprint arXiv:2101.00822},
  year={2021}
}
```

0. get source data ([WritingPrompts](https://github.com/pytorch/fairseq/blob/master/examples/stories/README.md), [WikiPlots](https://github.com/markriedl/WikiPlots)).
1. data pre-processing (data/), dataset statistics (dataset_statistics.py).
2. training (choose from several different implementations on parallelism: train.py, train_apex.py, train_dist.py, train_dist_apex.py).
3. generation, evaluation and analysis (generate.py, eval_ppl.py, generate_event_analysis.py).

Contact: lefang@buffalo.edu


Update on 2022:
If you encounter package version issue, sorry for that I don't have a requirements.txt with exact versions. I used this package: https://github.com/nvidia/apex and an old pytorch version compatible with it at that time, say pytorch=0.4 (not 100% sure).

