# Algorithmic Stability Based Generalization Bounds for Adversarial Training
-------
Code for the paper [Algorithmic Stability Based Generalization Bounds for Adversarial Training](https://openreview.net/pdf?id=2GwMazl9ND)

# To run the code

Run [python train.py](https://github.com/rzTian/AT-Stability/blob/main/train.py) to reproduce the experimental results in the paper.

For example, to reproduce the result of `tanh_{\gamma}-PGD AT` for `\gamma = 10` on CIFAR-10, run 
```
python train.py --dataset 'cifar10' --att_method 'tanh'  --beta 10
```

To reproduce the result of `G_{p}-PGD AT` for `p=2` on CIFAR-10, run

```
python train.py --dataset 'cifar10' --att_method 'norm_steep'  --beta 2
```

Experimental results will be saved in the folder [cifar10_results](https://github.com/rzTian/AT-Stability/tree/main/cifar10_results), [cifar100_results](https://github.com/rzTian/AT-Stability/tree/main/cifar100_results) or [svhn_results](https://github.com/rzTian/AT-Stability/tree/main/svhn_results)
