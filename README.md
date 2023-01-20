# Post-it segmentation

- Small project to fine-tune PyTorch's `fasterrcnn_resnet50_fpn` to specifically detect and segment images of post-it notes
- Intention to integrate with a handwritten text recognition model / API to create an easy way to transform real life image of post its into a digital task list
- Post-it training data is copied from https://github.com/valtech-uk/sticky-note-reader/tree/master/data

## Performance

### Before fine-tuning

![image](base_model/result_0.png)
![image](base_model/result_1.png)

### After fine-tining

![image](results/result_0.png)
![image](results/result_1.png)