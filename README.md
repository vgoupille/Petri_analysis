to do : 
- check isolation of petri dish on more images
- check split and rotation on more images for COV (attention create artefact on the edges due to rotation)
- rename each images : 
- convert all to jpg and 512*512 
- create a subdataset with all the pathogenes classes : one folder per class with all the images of this class
- data augmentation : rotation, flip, brightness, contrast, zoom
- train a yolo v8 seg model  for petri dish detection and segmentation
- verify the model on a validation set
- use the model to predict on the whole dataset
- need a correct annotation of the dataset (file or name convention

- use roboflow , labelme or napari for annotation
- Albumentations for data augmentation

https://albumentations.ai/docs/1-introduction/what-are-image-augmentations/

https://medium.com/@miramnair/image-and-label-augmentation-using-albumentations-b502a56bf486

https://youtu.be/Xhl_S_0ZYEo?si=uUXCB_CBHX_nyBmu