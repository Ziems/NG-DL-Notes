#### Types of Learning
- Supervised
	- *Structured/labeled data*
	- Ex: picture $\rightarrow$ label
	- Ex: picture $\rightarrow$ digit number
- Unsupervised
	- *Unstructured data*
	- Dataset of pictures(not labeled)

#### Train/Dev/Test Sets
- For small datasets, you have to allocate a larger % of examples for testing and validation. 
	- Ex: 70/30 train/test split
- For larger datasets, a much larger % of examples are used for testing.
	- Ex: 99/1/1 train/dev/test split
- Always make sure the dev and test sets come from the same distribution.

#### Bias/Variance
- High **Bias** means *underfitting*.
	- Ex: 15% training error and 16% dev error
- High **Variance** means *overfitting*.
	- Ex: 1% training error and 11% dev error.
- High bias and high variance can both be present.
	- Ex: 15% training error and 30 % dev error.
- In between high bias and high variance is **Just right**
	- Ex: 0.5% training error and 1% dev error.

#### Basic Recipe for Machine Learning
- Issues with high bias?
	- Increase the size of your network
	- Train longer
- High variance?
	- Try to get more data
	- Regularization
	- More appropriate architecture

#### Regularization
- **L2 Regularization**
	- *Sum of all the weights squared*
\begin{align*}
\lVert w \lVert_{2}^2 = \sum_{j=1}^{N_{x}}w_j^2
\end{align*}



