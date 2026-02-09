# Final Project

The project involves building a custom image dataset and implementing an object detection neural network.

* `/resources`: Resources for model training
	* `/cfg`: Store configuration files that control model settings, training parameters, and experiment setup
	* `/demo_2000`: Demo data (images and labels) and example outputs for testing
	* `images_test/`: Test images used for evaluation or inference
  * `.DS_Store`: macOS system file (can be ignored)
  * `yolo11n.pt`: Pretrained YOLOv11 model weights
* `/runs`: Model evaluation results
* `best.pt`: The pretrained model provided by instructor
* `notebook.ipynb`: Jupyter Notebook that shows evaluating code and running outcome
* `README.md`: [myself]()
* `report.pdf`: PDF version of the final report
* `train_10class.py`: Load and train the mode

## Model Details

- Model configuration: `resources/cfg/models/yolo11_model_10class.yaml`
- Trained model weights: `runs/detect/train/weights/last.pt` or `best.pt`
