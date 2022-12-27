To train stage II 

1. Extract the patches from WSI and the use the saved model weights from Stage I and extract the normalized features from the model.
2. Extract the corresponding coordinates from the WSI and normalize the coordinates based on the WSI shape.

The structure of the data patches should be like this.
<pre>
train/ 
    class_name1/
        wsi_1.npz
        wsi_2.npz
    class_name2/
        wsi3.npz
</pre>
and same for validation.

Each wsi_1.npz should have the following. <br>
features -> numpy of Nx128 features
coords -> numpy of Nx2 

Please see the data/sample folder for example.

To train run the following: 
```
 python train.py --data_path data/sample/ --num_graphs 5 --max-nodes 500 --sample_ratio=0.5 --load_data_list --input_feature_dim=18 --feature 'ca' --assign-ratio=0.10 --batch-size=1 --num_workers=0 --norm_adj --method='soft-assign' --lr=0.001 --step_size=10 --gcn_name='SAGE' --sampling_method='fuse' --g='knn' --drop=0.2 --jk
```
please customize as per your need.