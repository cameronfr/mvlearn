# A Co-Regularization Approach to Semi-supervised Learning with Multiple Views
### Vikas Sindwhani, Parth Niyogi, Mikhail Belkin, 2005
link: http://web.cse.ohio-state.edu/~belkin.8/papers/CASSL_ICML_05.pdf

## Abstract
Co-training bootstraps classifiers in each view. Here, they propose a co-regularization framework where classifiers are learnt in each view through forms of multi-view regularization. These algorithms are based on optimizing measures of agreement and smoothness over labeled and unlabeled examples. These algorithms extend standard regularization methods like SVM, and regularized least squares for MV semi-supervised learning.

## 1 Introduction
Co-training has become synonymous with a greedy agreement-maximization algorithm that is initialized by supervised classifiers in each view and then is iteratively re-trained on boosted labeled sets. 

**Family of algorithms:**

- Co-Regularized Least Squares (Co-RLS): joint regularization to minimize disagreement in a least square sense.
- Co-Regularized Laplacian SVM and Least Squares (Co-LapSVM, Co-LapRLS): use multi-view graph regularizers to enforce complementary and robust notions of smoothness in each view. (Use recently proposed Manifold Regularization techniques)

**Features:**

1. Natural extensions of classical RKHS framework
2. Non-greedy, involve convex cost functions
	* Whereas the greedy co-training algorithm never revises old classifications on unlabeled data that were added to the labeled pool
3. Influence of unlabeled data and multiple views can be controlled explicitly (single view semi-supervised and standard supervised learning are special cases of this framework)
4. Proposed methods outperform standard co-training

## 2 Multi-View Learning
In multi-view setting, where each example x = (x1, x2) is seen in two views, where x1 is in X1, and x2 is in X2. We want to learn the function pair f = (f1, f2). 

## 3 Co-Regularization
In co-regularization, attempt to learn the pair f = (f1, f2) in a cross-product of two RKHS defined over the two views.

**Co-Regularized Least Squares**

Try to learn the pair f = (f1, f2) s.t. each function correctly classifies the labeled examples, and the outputs of the pair agree over unlabeled examples. Suggests a certain objective function.
- Contains a parameter to balance data fitting in the two views, regularization parameters for the RKHS norms in the two views, and a coupling parameter that regularizes the pair towards compatibility using unlabeled data
	* When the coupling parameter = 0, system ignores unlabeled data and outputs supervised RLS solutions
- Can compute expansion coefficient vectors with a coupled linear system
- If you use a hinge loss, this can extend SVM in a similar manner

**Co-Laplacian RLS and Co-Laplacian SVM**

Want to learn the pair f = (f1, f2) s.t. each function correctly classifies labeled data and is smooth w.r.t. similarity structures in both views, which may be encoded as graphs over which regularization operators may be defined and then combined for MV.
- Construct a similarity graph for each view (s=1,2) whose adjacency matrix is W(s), where Wij(s) measures the similarity between xi(s), xj(s). The Laplacian matrix of this graph provides a smoothness functional on the graph.
One way to construct MV regularizer: take convex combination L = (1-a)L(1) + aL(2), which leads to a certain optimization problem.
