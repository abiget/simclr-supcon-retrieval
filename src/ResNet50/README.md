# My Project

This is a Python project that runs the ResNet50 model for image feature extraction.

## Contents

- `model_not_tuned.py`: runs not tuned ResNet50 and send submission JSON
- `fine_tuning.py`: fine-tune ResNet50 and save the updated weights as fine_tuned_model.pth
- `model_fine_tuned.py`: runs fine-tuned ResNet50 and send submission JSON
- `get_images.py`: includes utility functions used by both models
- `requirements.txt`: list of required Python packages

## How to Run

1. To run the model **without fine-tuning**, execute: python model_not_tuned.py

2. To **fine-tune the model**, first run: python fine_tuning.py

3. Then run the **fine-tuned model**: python model_fine_tuned.py

4. **Important**: Before running any script, make sure to **update the image folder paths** at the beginning of each file so they correctly point to your local training and testing directories.




