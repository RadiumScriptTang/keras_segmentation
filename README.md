# Keras_segmentation

In the offical version we found that the author was a little lazy :). In the section keras_segmentation.predict evaluate() function, the author defined function as following: 
```
def evaluate( model=None , inp_inmges=None , annotations=None , checkpoints_path=None ):

        assert False , "not implemented "

        ious = []
        for inp , ann   in tqdm( zip( inp_images , annotations )):
                pr = predict(model , inp )
                gt = get_segmentation_arr( ann , model.n_classes ,  model.output_width , model.output_height  )
                gt = gt.argmax(-1)
                iou = metrics.get_iou( gt , pr , model.n_classes )
                ious.append( iou )
        ious = np.array( ious )
        print("Class wise IoU "  ,  np.mean(ious , axis=0 ))
        print("Total  IoU "  ,  np.mean(ious ))
```

I bet the author was in a hurry and failed to finish this function. so let us finish it.

```
def evaluate( model=None , inp_images=None , annotations=None , checkpoints_path=None ):

	names = os.listdir(inp_images)
	images_annotations = [(os.path.join(inp_images,name),os.path.join(annotations,name)) for name in names]

	ious = []
	for inp , ann   in images_annotations:
		pr = predict(model , inp )
		gt = get_segmentation_arr( ann , model.n_classes ,  model.output_width , model.output_height  )
		gt = gt.argmax(-1)
		gt = gt.reshape(pr.shape)
		iou = metrics.get_iou( gt , pr , model.n_classes )
		ious.append( iou )
	ious = np.array( ious )
	print("Class wise IoU "  ,  np.mean(ious , axis=0 ))
	print("Total  IoU "  ,  np.mean(ious ))
  ```
  
  Thus the function can work.
