# SimCLR

This is a Pytorch implementation of the [SimCLR training procedure](https://arxiv.org/abs/2002.05709) by Chen et al. This was created to be as close to the original code and paper as possible, while still easy to use and understandable. Main features are :

1. Distributed training across multiple GPUs/TPUs
2. Synchronized batch normalization across all devices to prevent local information leakage (which could cause an improve of the prediction accuracy without improving the representations)
3. Thoroughly verified data augmentation policy
4. [LARS optimizer](https://arxiv.org/pdf/1708.03888.pdf), which was used in the original paper
5. Support for any image dataset
6. Coherent with both SimCLR v1 and SimCLR v2
