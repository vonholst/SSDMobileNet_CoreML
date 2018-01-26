# SSDMobileNet_CoreML
Real-time object-detection on iOS using CoreML model of SSD based on Mobilenet. This project contains an example-project for running real-time inference of that model on iOS.

- CoreML-file created following the example https://github.com/tf-coreml/tf-coreml/blob/master/examples/ssd_example.ipynb
- I used some of hollance convenient methods for object-detection (https://github.com/hollance/CoreMLHelpers), NMS/IoU/etc.
- The code for box-decoding from MobileNetV1-features was implemented by Vincent Chu (https://github.com/vincentchu).

To add the preprocessing to the coreml-model, I included some additional parameters in the convert-step (image_scale and bias). To make use of CoreVision's abilities to convert image-format / scaling, I added the image_input_names parameter:

```python
coreml_model = tfcoreml.convert(
      tf_model_path=frozen_model_file,
      mlmodel_path=coreml_model_file,
      input_name_shape_dict=input_tensor_shapes,
      image_input_names="Preprocessor/sub:0",
      output_feature_names=output_tensor_names,
      image_scale=2./255.,
      red_bias=-1.0,
      green_bias=-1.0,
      blue_bias=-1.0
)
```
