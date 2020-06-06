# Models

FalconCV offers a interface to work with the models from the Tensorflow Object Detection API [^1]
[^1]: https://github.com/tensorflow/models

## Load a pre-trained model

| parameter  | description                      |
| ---------- | -------------------------------- |
| model      | path to saved model              |
| labels_map | path to the labels map (*.pbtxt) |

!!! note "Load Saved model"
    ````python
    from falconcv.models import ModelBuilder
    from falconcv.util import VIUtil
    import falconcv as fcv

    if __name__ == '__main__':        
        # load saved model
        with ModelBuilder.build("<path to saved model>", "<path to the labels map>") as model:
            img, predictions=model.predict("<image path | image uri >")
            VIUtil.imshow(img,predictions)
    ````

| parameter  | description                      |
| ---------- | -------------------------------- |
| model      | path to the freeze model (*.pb)  |
| labels_map | path to the labels map (*.pbtxt) |

!!! note "Load Freeze model"
    ````python
    from falconcv.models import ModelBuilder
    from falconcv.util import VIUtil
    import falconcv as fcv

    if __name__ == '__main__':        
         # load freeze model
        with ModelBuilder.build("<path to the freeze model *.pb>", "<path to the labels map>") as model:
            img, predictions=model.predict("< image path | image uri >")
            VIUtil.imshow(img,predictions)
    ````

## Train Custom model

**Build:**

| parameter  | description                                |
| ---------- | ------------------------------------------ |
| config     | dictionary with configuration for training |

**Train:**

| parameter         | description                           | example                                  |
| ----------------- | ------------------------------------- | ---------------------------------------- |
| epochs            | number of epochs                      | epochs=200 (default: 100)                |
| val_split         | split percentage for val              | val_split=0.2 (default: 0.3)             |
| clear_folder      | clear folder                          | clear_folder=False (default: False)      |
| override_pipeline | override the pipeline file            | override_pipeline=False (default: False) |
| eval              | indicates if make eval while training | eval=True (default: False)               |

!!! note "Train Custom model"
    ````python
    from falconcv.models import ModelBuilder
    from falconcv.util import VIUtil
    import falconcv as fcv

    if __name__ == '__main__':
        config = {
            "model": model_name,
            "images_folder": images_folder,
            "output_folder": out_folder,
            "labels_map": labels_map,
        }
        with ModelBuilder.build(config=config) as model:
            model.train(epochs=epochs,val_split=0.3, clear_folder=False)
            # for freezing the model            
            model.freeze(chekpoint=epochs)
    ````

!!! tip "List TFODAPI models zoo"
    ```python
    from falconcv.models.tf import ModelZoo

    if __name__ == '__main__':
        # only faster_rcnn
        ModelZoo.print_available_models(arch="faster")
    
        # print all
        ModelZoo.print_available_models()
    ```
