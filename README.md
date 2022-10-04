# Event-based Dynamic Graph Representation Learning for Patent Application Trend Prediction
## Datasets
We provide the preprocessed datasets at [here](https://drive.google.com/drive/folders/10SPt1y2bGpywosbMce9oRHGBsJGMlUpY?usp=sharing), which should be put in the ./data folder.

## To run the patent application trend prediction task on patent datasets:
run ./main.py to get the results of EDGPAT.

## Performance
|Metric|Recall-10|DNCG-10|PHR-10|Recall-20|DNCG-20|PHR-20|Recall-30|DNCG-30|PHR-30|Recall-40|DNCG-40|PHR-40|
|----|----|----|----|----|----|----|----|----|----|----|----|----|
|EDGPAT|0.1175|0.1725|0.5491|0.1646|0.1742|0.6304|0.1868|0.1769|0.6612|0.2006|0.1769|0.6800|

## Environments:
* [PyTorch 1.7.1](https://pytorch.org/)
* [tqdm](https://github.com/tqdm/tqdm)
* [numpy](https://github.com/numpy/numpy)
