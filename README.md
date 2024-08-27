# AEO
Based on the paper https://arxiv.org/abs/2402.14031

by M. Augustine, P. Patil, M. Bhushan and S. Bhartiya

This work presents an autoencoder with ordered
variance (AEO) which is based on a modified loss function with a
variance regularization term that orders the latent variables 
by decreasing variance.   The autoencoder is further modified
using ResNets, which results in a ResNet AEO (RAEO). The AEO and RAEO
can be used for extracting nonlinear relationships in unlabeled
dataset which leads to unsupervised static model extraction. 
