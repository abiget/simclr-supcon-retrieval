
import json
# calculate top-k accuracy
def calculate_top_k_accuracy(path_to_submission, top_k=5):
    """
    Calculate the top-k accuracy from a submission file and return the overall accuracy and per-class accuracy.
    Args:
        path_to_submission (str): Path to the submission JSON file.
        top_k (int): The number of top predictions to consider for accuracy calculation.
    Returns:
        tuple: Overall accuracy and a dictionary with per-class accuracy."""

    with open(path_to_submission) as f:
        submission = json.load(f)
        
    if not isinstance(submission, list):
        raise ValueError("Submission must be a list")

    class_correct = {}
    class_total = {}
    correct = 0
    for item in submission:
        label = item['label']

        if label not in class_correct:
            class_correct[label] = 0
            class_total[label] = 0

        if label in [match.split('/')[-2] for match in item['samples'][:top_k]]:
            correct += 1
            class_correct[label] += 1
        class_total[label] += 1
    
    accuracy_per_class = {cls: (class_correct[cls] / class_total[cls]) * 100 for cls in class_correct}
    accuracy = correct / len(submission) * 100

    return accuracy, accuracy_per_class