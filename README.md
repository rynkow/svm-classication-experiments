# SVM classication experiments
Experiments on 2d datasets with various SVM types.

## Experiments
### Used datasets
I prepared a dataset, that allows for good linear classification, but should work better with nonlinear kernels. <br><br>
![data](experiment_results/dataset.png)

### Linear SVM
#### Kernel function: K(x, x') = <x, x'>
![rbf](experiment_results/linearSVM.png)


### RBF kernel SVM
#### Kernel function: K(x, x') = exp(-gamma*||x-x'||^2)
![rbf](experiment_results/rbfSVM.png)

### Polynomial kernel SVM
#### Kernel function: K(x, x') = (gamma*<x, x'> + coef)^dim
![polynomial](experiment_results/polySVM.png)

### Custom kernel SVM
#### Kernel function: K(x, x') = (R - ||x - x'||)/R if ||x - x'|| <= R, else 0
<img src="experiment_results/customSVM.png" alt="custom" width="700"/>

