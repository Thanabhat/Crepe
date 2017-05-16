# Dynamic Character-level Convolutional Neural Network

This repository contains code in Torch 7 for my research paper

Thanabhat Koomsubha and Peerapon Vateekul [A character-level convolutional neural network with dynamic input length for Thai text categorization](http://ieeexplore.ieee.org/document/7886102/)

**Note**: Data in this repository is not real data that we used in our research.

This project is forked from [Crepe](https://github.com/zhangxiangxiao/Crepe)

## Example Usage

Prerequisite
- Torch 7 with CUDA installed

Extract `data.tar.gz` and build `t7b` file
```sh
$ cd data
$ tar -xvf data.tar.gz
$ qlua csv2t7b.lua -input train.csv -output train.t7b
$ qlua csv2t7b.lua -input valid.csv -output valid.t7b
$ qlua csv2t7b.lua -input test.csv -output test.t7b
$ cd ..
```

Run 
```sh
$ cd train
$ qlua main.lua
```
