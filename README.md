# VarianceReducedSG

[![Build Status](https://travis-ci.org/mvmorin/VarianceReducedSG.jl.svg?branch=master)](https://travis-ci.org/mvmorin/VarianceReducedSG.jl)
[![codecov](https://codecov.io/gh/mvmorin/VarianceReducedSG.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mvmorin/VarianceReducedSG.jl)

#### Work in progress

## Introduction
The main goal of this package is to provide an interface for calculating variance-reduced gradients on the form

<img src="https://latex.codecogs.com/gif.latex?\inline&space;G_i(x,y)&space;=&space;\theta&space;(\nabla&space;f_i(x)&space;-&space;y_i)&space;&plus;&space;\frac{1}{n}&space;\sum_{i=1}^n&space;y_i" title="G_i(x,y) = \theta (\nabla f_i(x) - y_i) + \frac{1}{n} \sum_{i=1}^n y_i" />.

The most efficient way of implementing this update and storing the dual variables *y_i* depends on the properties of *f_i*. The provided interface and implementations of different variance reduced gradient makes it possible to write general **Variance-Reduced Stochastic Gradient** algorithms without considering the storage of the dual variables.

Alongside the variance-reduced gradients the package also provide a solver with implementations of different variance-reduced algorithms (SAGA, SVRG, etc.) and an interface for logging of convergence history. Note, the focus of the solver is not to provide an easy to use interface for formulating and solving optimization problems, rather, it is geared towards the ease of implementation and the introspection of variance-reduced algorithms.

The solver and the algorithm implementations might not remain a permanent part of this package and might be moved to/replaced by other packages.

## Usage
ToDo

## Examples
ToDo
