# retinal-vessel-segmentation

[S-UNet: A Bridge-Style U-Net Framework with a Saliency Mechanism for Retinal Vessel Segmentation](http://sci-hub.tw/https://ieeexplore.ieee.org/document/8842560)

# Dataset

[DRIVE](http://www.isi.uu.nl/Research/Databases/DRIVE/) [CHASE_DB1](https://blogs.kingston.ac.uk/retinal/chasedb1/) [TONGREN](http://111.zbj99.cn/list.php?pid=3)

# S-UNet

![](https://github.com/huawang123/retinal-vessel-segmentation/blob/master/img/GA.png) 

# Results
 
|Dataset | Method        | Year  | MCC    | SE     | SP     | ACC    | AUC    | F1-scores | Patch/Image-based|
|----  | ----        |----  | ----  | ----   | ----   | ----   | ----   | ----      |---- |
|[DRIVE](http://www.isi.uu.nl/Research/Databases/DRIVE/)   | S-UNet  | 2019  | 0.8055 | 0.8312 | 0.9751 | 0.9567 | 0.9821 | 0.8303    | Image-based|
|[CHASE_DB1](https://blogs.kingston.ac.uk/retinal/chasedb1/) | S-UNet |2019|0.8065|0.8044|0.9841|0.9658|0.9867|0.8242|Image-based|
|[TONGREN](http://111.zbj99.cn/list.php?pid=3)|S-UNet |2019|0.7806|0.7822|0.9830|0.9652|0.9824|0.7994|Image-based|

The details in DRIVE:

![](https://github.com/huawang123/retinal-vessel-segmentation/blob/master/img/as.jpg)  

# Quickstart

1. Download and extract [DRIVE](http://www.isi.uu.nl/Research/Databases/DRIVE/) file in fold './data'
2. Training 

    ```python3 main.py --data_type=DRIVE --train=True --Restore=False```

2. Test

    ```python3 main.py --data_type=DRIVE --train=False --Restore=True --load_checkpoint=log_dir```

The CHASE_DB1 is same with DRIVE. but the TONGREN needs to convert png format from png use convert.py firstly.


# Citation

If the code is helpful for your research, please consider citing:

>10.1109/ACCESS.2019.2940476

# Questions

Please contact ‘hwang0609@163.com‘
