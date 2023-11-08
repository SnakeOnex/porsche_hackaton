# E2E learning for Porsche Hackaton

## Concept
Collect a dataset of (image, steering_angle) pairs and train a conv-net to predict the steering angle from the image.

### Data format
- data/ folder, with <name>.jpg as the image and <name>.txt for the steering angle.
- Tested on [Sample dataset](https://www.kaggle.com/datasets/zahidbooni/alltownswithweather/):
    - just unzip it in the project root and use the 'load_from_kaggle' dataset function (default)
    - [Related blogpost](https://imtiazulhassan.medium.com/end-to-end-learning-using-carla-simulator-12869b5d6f7)

## TODO
- [X] dataset class
- [X] inference code
- [X] training loop
- [ ] data augmentation to insentivise recovery behavior

