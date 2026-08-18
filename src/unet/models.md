
Exploramos cada cambio de UNET++ en un notebook aparte porque lleva mucho tiempo de entrenamiento. 

### 5 Training
* [unet_plus_plus_resnet18_5layers.ipynb](./training/unet_plus_plus_resnet18_5layers.ipynb) — *ResNet18 (No scSE)*
* [unet_plus_plus_resnet18_5layers_scse.ipynb](./training/unet_plus_plus_resnet18_5layers_scse.ipynb) — *ResNet18 (With scSE)*
* [unet_plus_plus_resnet34_5layers.ipynb](./training/unet_plus_plus_resnet34_5layers.ipynb) — *ResNet34 (No scSE)*
* [unet_plus_plus_resnet34_5layers_scse.ipynb](./training/unet_plus_plus_resnet34_5layers_scse.ipynb) — *ResNet34 (With scSE)*
* [unet_plus_plus efficientnet_5layers.ipynb](./training/unet_plus_plus%20efficientnet_5layers.ipynb) — *EfficientNet (No scSE)*
* [unet_plus_plus_efficientnet_5layers_scse.ipynb](./training/unet_plus_plus_efficientnet_5layers_scse.ipynb) — *EfficientNet-B0 (With scSE)*

Benchmarks para los 3 mejores modelos::
* [Benchmark: Unet++ EfficientNet-B0 5layers scSE](./benchmark_unet_5_layers_efficientnet_5layers_scse.ipynb) 
* [Benchmark: Unet++ ResNet34 5layers scSE](./benchmark_unet_5_layers_resnet34_5layers_scse.ipynb)
* [Benchmark: Unet++ ResNet34 5layers](./benchmark_unet_5_layers_resnet34_5layers.ipynb) 