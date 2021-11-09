# MUSIC MOOD CLASSIFICATION USING CONVOLUTIONAL NEURAL NETWORKS

By Eliott Remmer & Karl Simu

## Introduction

Mood recognition of a piece of music is a common topic of research within Music Informatics Retrival (MIR). There has been much interest in the field in recent years by the advance of music streaming services as well as the rise of related fields such as Music Recommendation Systems . One of the prominent approaches to mood recognition of music have been automatic tagging. In this task, one aims to predict the music tags, i.e. the high-level mood information of a music clip. The problem is further formalized as a multi-label classification problem where a track is assigned one or more fixed mood tags. In this project, we have used CNN model architectures presented in a paper by Choi et. al. [1] to try to improve baseline results using the MTG-Jamendo dataset [2] for mood/theme classification to music. The code of this project mainly works within a script framework already presented with the MTG-Jamendo dataset.

## Structure

Most of the files in this repo comes from the [MTG-jamendo Dataset repo](https://github.com/MTG/mtg-jamendo-dataset). While some folders & files have been removed in this version, many are kept for structure & transperency resons. The old README is found under old-readme/README.md

The models can be found in:

* scripts/baseline/fcn5.py
* scripts/baseline/fcn6.py
* scripts/baseline/fcn7.py

All scripts written or modified by us:
* scripts/baseline/fcn5.py
* scripts/baseline/fcn6.py
* scripts/baseline/fcn7.py
* scripts/baseline/data_loader.py
* scripts/baseline/get_npy.py
* scripts/baseline/solver.py
* scripts/specplot.py
* AUC-Loss-plots/plot.py


## Refrences

[1] Keunwoo  Choi,  Gy ̈orgy  Fazekas,  and  Mark  B.  San-dler. Automatic tagging using deep convolutional neu-ral networks.CoRR, abs/1606.00298, 2016.

[2] Bogdanov, D., Won M., Tovstogan P., Porter A., & Serra X. (2019).  [The MTG-Jamendo Dataset for Automatic Music Tagging](https://hdl.handle.net/10230/42015). Machine Learning for Music Discovery Workshop, International Conference on Machine Learning (ICML 2019).


## License

* The code in this repository is licensed under [Apache 2.0](LICENSE) 
* The metadata is licensed under a [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
* The audio files are licensed under Creative Commons licenses, see individual licenses for details in `audio_licenses.txt`.

## Acknowledgments

2019 Music Technology Group For providing the dataset and additional scripts
