
## Introduction

This code is the implement of our XXX 2020 paper (under review). Our model is based on [mt-dnn](https://github.com/namisan/mt-dnn). The main difference is the sampling and training strategy used in this paper, which is the file `./mt_dnn/acl_controller.py`.

## Preparation
The glue experiments are defined in `experiments/glue`, while the transfer learning experiments are defined in `experiments/transfer`

Environment setups:
```bash
pip install -r requirements.txt
```



## Data Preprocessing

Download GLUE data
```bash
sh download.sh
```

Please refer to download GLUE dataset: https://gluebenchmark.com/

Preprocess Glue data
```bash
sh experiments/glue/prepro.sh
```


Download transfer learning data
```bash
TODO
```

Preprocess data
```bash
sh experiments/transfer/prepro.sh
```



## Train Model

Train model
```bash
sh scripts/acl_controller.sh 0.5
```
where the first argument is the `\phi` value in our policy


## Transfer Learning

    TODO


## Citation

    TODO

