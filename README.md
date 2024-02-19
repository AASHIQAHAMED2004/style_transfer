# Neural Style Transfer with VGG19

Welcome to the Neural Style Transfer project using VGG19! This program aims to transfer the artistic style of one image to the content of another image using convolutional neural networks (CNNs) and optimization techniques.

## Overview

Neural style transfer is a technique that combines the content of one image with the style of another image to create a new image that preserves both content and style characteristics. In this project, we use the VGG19 model as a feature extractor to capture the content and style information of the input images.

## How it Works

1. **Data Preparation**: Ensure that the content image (e.g., "mona_lisa.PNG") and the style image (e.g., "hex.jpg") are available in the same directory as the script.

2. **Model Initialization**: The VGG19 model is initialized and configured to extract features from specific layers chosen for style transfer.

3. **Image Loading**: The content and style images are loaded, resized, transformed into tensors, and moved to the appropriate device (GPU if available).

4. **Optimization Setup**: Adam optimizer is set up to optimize the generated image to minimize the content and style loss.

5. **Training Loop**: The main training loop iterates for a specified number of epochs. In each iteration, the content and style losses are computed using features extracted from the VGG19 model. The total loss, which is a combination of content and style losses, is minimized through backpropagation.

6. **Output**: The generated image is saved periodically during training. The user can monitor the total loss to gauge the progress of style transfer.

## Requirements

- PyTorch
- torchvision
- PIL (Python Imaging Library)

## Usage

1. Ensure that you have the required dependencies installed. You can install them using pip:

2. Run the script in a Jupyter Notebook or a Python environment. Make sure to have the content and style images in the same directory and adjust any paths if necessary.

3. Monitor the training process, and after completion, you will find the generated image saved as "generated.png" in the same directory.

## Customization

- Feel free to experiment with different content and style images to create unique combinations.
- Adjust hyperparameters such as learning rate, alpha, and beta to achieve desired results.
- Explore different layers of the VGG19 model for style extraction to influence the artistic style of the generated image.

Happy style transferring!
