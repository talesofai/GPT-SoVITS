#!/bin/bash

# 删除GPT_weights文件夹下的所有文件
rm -f GPT_weights/*

# 删除SoVITS_weights文件夹下的所有文件
rm -f SoVITS_weights/*

rm -rf logs/*

# 删除output文件夹下的所有文件（不包括文件夹）
find output/ -type f -print -delete

