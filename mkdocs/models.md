# Models

FalconCV offers a interface to work with the models from the Tensorflow object detection API [^1]
[^1]: https://github.com/tensorflow/models

## Load a pre-trained model

!!! note "Load Saved model"
    ````python
    from falconcv.models import ModelBuilder
    from falconcv.util import VIUtil
    import falconcv as fcv    
    if __name__ == '__main__':        
        # load saved model
        with ModelBuilder.build("<path to saved model>","<path to the labels map>") as model:
            img,predictions=model.predict("<image path | image uri >")
            VIUtil.imshow(img,predictions)
    ````

!!! note "Load Freeze model"
    ````python
    from falconcv.models import ModelBuilder
    from falconcv.util import VIUtil
    import falconcv as fcv    
    if __name__ == '__main__':        
         # load freeze model
        with ModelBuilder.build("<path to the freeze model *.pb>","<path to the labels map>") as model:
            img,predictions=model.predict("< image path | image uri >")
            VIUtil.imshow(img,predictions)
    ````
    
## Train Custom model

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


!!! tip "List models zoo"
    ```python
    from falconcv.models.tf import ModelZoo
    if __name__ == '__main__':
        # only faster_rcnn
        model_zoo = ModelZoo.available_models(arch="faster")
        for model_name, model_uri in model_zoo.items():
            print(model_name, model_uri)
    
        # print all
        model_zoo=ModelZoo.available_models()
        for model_name,model_uri in model_zoo.items():
            print(model_name,model_uri)
    ```




