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

            
