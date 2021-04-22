# Utilisation de ce repo

python train.py --datadir /path/to/data/ --pretrained --frozen --model_name resnet50 --epochs 100 --batch_size 256

Le `datadir` doit être comme suit : 

* `/path/to/data/`:
    * `train/`:
        * `label_0/`:
            * `image_0.jpg`
            * `ìmage_1.jpg`
            * ...
        * `label_1/`:
            * `ìmage_0.jpg`
            * ...
        *  ..
    * `val/`:
        * ...
            * ...

Pour plug un autre réseau, le construire à coté, l'ajouter dans le dictionnaire de la méthode `.get_network` avec comme clef 
la valeur rentrée dans l'argument `--model_name`.

# Results of the classification task on PCam

|              | Flips              | GaussianBlur       | Rotate90           | HEaug              | Jitter             | ElasticDistorsion  | DifferentCrops     | CropAndResize      | N°epochs for MoCo  |
|--------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| Jitter       | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | 200                |
| Jitter400    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | 400                |
| HEaug        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                | 200                |
| SameCrop     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | 200                |
| ResizeCrop   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | 200                |
| GaussianBlur | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | 200                |
| Tristan      | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: |
| Baseline     | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: |
| Random       | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: |
