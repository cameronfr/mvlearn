# Combining Labeled and Unlabeled Data with Co-Training
### Avrim Blum & Tom Mitchell, 1998
link: https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf

## 1 Introduction
- We want to tackle the common problem where we have many unlabeled examples, but much fewer labeled.
- Many of these are cases where there are multiple views of the data, where one of them is labeled and the other is not.
	* Ex. Web page classification
		Views: 1) Text on the page 2) Words in hyperlinks that point to that page
	* These views may be partially redundant, but they can aid each other in training better classifiers

## 2 A Formal Framework
- X = X_1 cross X_2, where X_1 and X_2 are two different views, i.e. an example x = (x_1,x_2).
- Assume that the two views agree, and produce the same labels
- Let D be the distribution over X
- Let f_1, f_2 be target functions for X_1 and X_2
- Assume that D assigns probability zero to any examples (x_1,x_2) s.t. f_1(x_1) != f_2(x_2).


## 3 A High Level View and Relation to Other Approaches
Imagine one view (labeled) would produce a set of possible distributions C if used alone. By using the unlabeled data, we hope to reduce this to a smaller set C' of functions in C that are also compatible with what is known about D (the distribution over X).

## 4 Rote Learning
By having two views, it is potentially allowable to have many fewer labeled examples if we have a large number of unlabeled samples, since we will still have data from a large part of the sample space. (The paper contains a formal proof and framework for this)

## 5 PAC Learning in Large Input Spaces
"Given a conditional independence assumption on the distribution D, if the target class is learnable from random classification noise in the standard PAC model, then any initial weak predictor can be boosted to arbitrarily high accuracy using only unlabeled examples by co-training."
- x_1 and x_2 are conditionally independent given the label
	- Ex. the words on a page P and the words on hyperlinks that point to P are independent of each other when conditioned on the classification of P
Lots of proofs

## 6 Experiments
Tried to classify web pages. Specifically, target function was the home page of a CS department at some school where web pages we taken from. These were positive examples, and negative examples were just any web pages. 
- x_1 is the bag (multi-set) of words in the web page
- x_2 is the bag of words underlined in all links pointing to this web page from other pages
- Naive Bayes classifiers were trained separately for x_1 and x_2

**Co-training algorithm**

Given
- a set L of labeled training examples
- a set U of unlabeled examples

Create a pool U' of examples by choosing u examples at random from U

Loop for k iterations:
- Use L to train a classifier h_1 that considers only the x_1 portion of x
- Use L to train a classifier h_2 that considers only the x_2 portion of x
- Allow h_1 to label p positive and n negative examples from U'
- Allow h_2 to label p positive and n negative examples from U'
- Add these self-labeled examples to L
- Randomly choose 2p + 2n examples from U to replenish U'

*Note:* The paper experiments use naive Bayes classifiers for both h's, p=1, n=3, k=30, u=75, size(U) = 1051, size(L) = 12

***Final Combined Classifier***

Since naive Bayes classifiers were used, which output probabilities P(c_j|x_1) of class c_j given the instance x_1, and P(c_j|x_2) of class c_j given the instance x_2, the combined classifier uses the conditional independence assumption, and outputs P(c_j|x) of class c_j given the instance x = (x1,2) as *P(c_j|x) = P(c_j|x_1)P(c_j|x_2)* 

