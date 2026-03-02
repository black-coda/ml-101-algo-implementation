## Support Vector Machine (SVM) Technical Report

Support Vector Machines (SVM) represent a class of supervised learning models used for classification and regression analysis. This report outlines the mathematical construction of the SVM, from the basic linear classifier to the application of the kernel trick for non-linear datasets.

---

### 1. The Foundation of Binary Classification

The objective of binary classification is to separate a labeled dataset  using a decision boundary. In this context:

*  represents the feature vector.
*  denotes the class label.

The classification function is defined as , where  is the weight vector and  is the bias term.

---

### 2. The Linear Hyperplane

A hyperplane in an -dimensional space is defined by the equation . This boundary partitions the space into two regions:

* Class +1: 
* Class -1: 

Unlike simple linear classifiers, SVM specifically seeks a hyperplane that optimizes for generalization by maximizing the margin.

---

### 3. Maximum Margin Principle

The margin is defined as the distance between the hyperplane and the nearest data points from either class, known as support vectors. The geometric distance from any point  to the hyperplane is calculated as:

By normalizing the functional margin such that  for all points, the total margin width becomes . Consequently, maximizing the margin is mathematically equivalent to minimizing the reciprocal expression:

---

### 4. Hard-Margin SVM Optimization

In cases where data is perfectly linearly separable, the optimization problem is formulated as a constrained quadratic programming problem:

This approach ensures strict separation with the maximum possible margin, though it is highly sensitive to outliers and requires perfectly separable data.

---

### 5. Soft-Margin SVM and Slack Variables

To handle non-linearly separable or noisy data, the constraints are relaxed using slack variables . This allows for certain data points to fall within the margin or be misclassified. The objective function is modified to include a regularization parameter :

The parameter  determines the trade-off between maximizing the margin and minimizing classification errors. A high  value penalizes misclassifications heavily, leading to a narrower margin.

---

### 6. The Dual Formulation and Kernel Methods

The optimization problem can be solved in its Dual Form using Lagrange multipliers :

The introduction of the kernel function  allows the model to compute dot products in high-dimensional feature spaces without explicitly performing the transformation. Common kernels include:

* **Linear:** 
* **Polynomial:** 
* **Radial Basis Function (RBF):** 

---

### 7. Final Decision Function and Hinge Loss

The final decision function depends only on the support vectors (points where ):

This framework essentially minimizes the Hinge Loss, , which provides the SVM its characteristic robustness against points far from the decision boundary.

---