# BoVW-CN

The BoVW-CN method is implemented in this repository, which combines **Bag-of-Visual-Words and complex networks** for describing keypoints detected in a given image.

In this project it is possible to use the detectors of points of interest SIFT, SURF, FAST, ORB, HARRIS, RANDOM, DENSE, STAR, BRISK, BLOB and GFTT, the descriptors of points of interest SIFT, SURF, BOC, LBP, Fourier, BIC and Networks, in addition to building the vocabulary of visual words with the random, k-means and unsupervised OPF method.

## Requirements:
 - OpenCV 2.4.10
 - LibOPF
 - iGraph

## Citation
If you use this methodology in a scientific publication, please cite with the following DOI:
```
@article{DELIMA2019215,
        title = "Classification of texture based on Bag-of-Visual-Words through complex networks",
        journal = "Expert Systems with Applications",
        volume = "133",
        pages = "215 - 224",
        year = "2019",
        issn = "0957-4174",
        doi = "https://doi.org/10.1016/j.eswa.2019.05.021",
        url = "http://www.sciencedirect.com/science/article/pii/S0957417419303483",
        author = "Geovana V.L. de Lima and Priscila T.M. Saito and Fabricio M. Lopes and Pedro H. Bugatti",
        keywords = "Classification, Texture, Bag-of-visual words, Complex networks",
}
```
