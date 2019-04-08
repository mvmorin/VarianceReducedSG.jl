# VarianceReducedSG

[![Build Status](https://travis-ci.org/mvmorin/VarianceReducedSG.jl.svg?branch=master)](https://travis-ci.org/mvmorin/VarianceReducedSG.jl)
[![codecov](https://codecov.io/gh/mvmorin/VarianceReducedSG.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mvmorin/VarianceReducedSG.jl)

#### Work in progress

## Introduction
The main goal of this package is to provide an interface for calculating variance-reduced gradients on the form

<img src="https://latex.codecogs.com/gif.latex?\inline&space;G_i(x,y)&space;=&space;\frac{\theta}{n}&space;(\nabla&space;f_i(x)&space;-&space;y_i)&space;&plus;&space;\frac{1}{n}&space;\sum_{i=1}^n&space;y_i" title="G_i(x,y) = \frac{\theta}{n} (\nabla f_i(x) - y_i) + \frac{1}{n} \sum_{i=1}^n y_i" />.

The most efficient way of storing the dual variables y_i depends on the problem properties and the most efficient implementation therefore change depending on problem. The provided interface and the several implementations of the variance reduced gradient makes it possible to write general **Variance-Reduced Stochastic Gradient** algorithms without considering the storage of the dual variables.

Alongside the variance-reduced gradients the package also provide a solver with implementations of different variance reduced algorithms and an interface for logging of convergence history. Note, the focus of the solver is not to provide an easy to use interface for formulating and solving optimization problems, rather, it is geared towards the ease of implementation and the introspection of variance-reduced algorithms.

## Usage
ToDo

## Examples
ToDo
