## Code
### 1. Generate the Synthetic Dataset
Using Stable Diffusion v1.4, latent consistency model, PixArt-$\alpha$ or Stable Cascade to Generate the Synthetic Dataset:
#### 1.1 Generate CIFAR10/CIFAR100:
    ```
    python create_dataset.py \
        --dataset <Dataset Name> \
        --outdir <PATH/TO/DATASET/DIR> \
        --model_type <Text-to-image Model Type> \
        --num_imgs <The Sample Size of Synthetic Dataset> \
        --dataset_to_gen <Generate the Training Set of Suspicious Model, Shadow Dataset or Validation Dataset> \
        --gpu <GPU ID>
    ```

#### 1.2 Generate ImageNet-100:
    ```
    python create_imagenet_dataset.py \
        --outdir <PATH/TO/DATASET/DIR> \
        --model_type <Text-to-image Model Type> \
        --num_imgs <The Sample Size of Synthetic Dataset> \
        --dataset_to_gen <Generate the Training Set of Suspicious Model, Shadow Dataset or Validation Dataset> \
        --gpu <GPU ID>
    ```

### 2. Train the Suspicious Model / Shadow Model:
```
python train_network.py \
    --model <Model Architecture> \
    --batch_size <Batch Size> \
    --lr <Learning Rate> \
    --wd <Weight Decay> \
    --loss <Loss Function> \
    --epoch <Training Epoch> \
    --combine <Whether to Train Using Both the Real Dataset and the Synthetic Dataset Together> \
    --shadow <Whether to Train the Shadow Model> \
    --dataset_size <The Proportion of the Original Dataset Used for Training> \
    --gpu <GPU ID>
```

### 3. Using TrainProVe to Verify the Training Data Provenance of the Suspicious Model

```
python eval_all.py \
    --model_root <The Root Directory for Storing Models> \
    --dataset <Dataset Name> \
    --positive <Whether to Detect Positive Examples> \
    --criterion <Use Accuracy/Entropy/Binary Cosine Similarity as the Detection Criterion (TrainProVe/TrainProVe-Ent/TrainProVe-Sim)> \
    --G_d <The Text-to-image Model of Defender> \
    --G_sus <The Data Source of Suspicious Model> \
    --arch_shadow <The Architecture of the Shadow Model> \
```

## Acknowledgement
This resipotry is based on [Diversity is Definitely Needed](https://github.com/Jordan-HS/Diversity_is_Definitely_Needed)

## Citation
If you find our data or project useful in your research, please cite:
```
@article{xie2025training,
  title={Training Data Provenance Verification: Did Your Model Use Synthetic Data from My Generative Model for Training?},
  author={Xie, Yuechen and Song, Jie and Wang, Huiqiong and Song, Mingli},
  journal={arXiv preprint arXiv:2503.09122},
  year={2025}
}
```
