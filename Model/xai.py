import os
import torch
import numpy as np
from Model.FinLangNet import MyModel_FinLangNet

# Ensure deterministic behavior in CUDA operations, necessary for debugging and reproducibility.
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Automatically select device based on availability of CUDA.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_feature_importance(model, input_features, all_inputs, non_evaluated_features_indices, epsilon=1, num_iterations=10):
    """
    Evaluates the importance of feature groups in the context of a multi-label problem considering all model inputs.

    It perturbs the input features by a small epsilon to assess the sensitivity of the model's output, thereby
    estimating the importance of each feature group and their sub-features.

    Args:
        model (torch.nn.Module): The pre-trained model to evaluate.
        input_features (list of torch.Tensor): The list of input feature groups to evaluate, each with shape
                                               (batch_size, num_features, sequence_length).
        all_inputs (list of torch.Tensor): The complete list of model inputs, including evaluated
                                           and non-evaluated features.
        non_evaluated_features_indices (list of int): Indices in `all_inputs` that correspond to features not to be evaluated.
        epsilon (int, optional): The perturbation coefficient. Defaults to 1.
        num_iterations (int, optional): The number of iterations for the perturbation analysis. Defaults to 10.

    Returns:
        dict: A dictionary mapping from feature group names to a list of (sub-feature index, importance score) tuples,
              sorted by importance score in descending order.
    """
    model.eval()  # Set model to evaluation mode.
    original_output = model(*all_inputs)[1]  # Original model output without any perturbation.

    # Map input_features to their corresponding indices in all_inputs.
    evaluated_features_indices = [i for i in range(len(all_inputs)) if i not in non_evaluated_features_indices]
    feature_importances = {}

    for eval_index, feature_index in enumerate(evaluated_features_indices):
        feature = input_features[eval_index]  # Extract feature group to be evaluated.
        feature_importance = []

        # Evaluate importance for each sub-feature within the feature group.
        for sub_feature_idx in range(feature.shape[1]):
            differences_sum = 0  # Initialize sum of output differences due to perturbations.

            for _ in range(num_iterations):  # Apply perturbation num_iterations times.
                perturbed_feature = feature.clone()
                # Generate perturbation.
                int_perturbation = torch.randint(low=-epsilon, high=epsilon + 1, 
                                                 size=perturbed_feature[:, sub_feature_idx, :].shape, 
                                                 dtype=torch.int64).to(device)
                perturbed_feature[:, sub_feature_idx, :] += int_perturbation
                perturbed_feature[:, sub_feature_idx, :] = torch.clamp(perturbed_feature[:, sub_feature_idx, :], min=1, max=15)
                
                # Prepare perturbed inputs for the model.
                perturbed_inputs = list(all_inputs)
                perturbed_inputs[feature_index] = perturbed_feature  # Replace the original feature with its perturbed version.

                perturbed_output = model(*perturbed_inputs)[1]  # Get model output with perturbed input.
                differences_sum += (perturbed_output - original_output).abs().mean().item()  # Accumulate the output difference.

            average_difference = differences_sum / num_iterations  # Compute the average difference per sub-feature.
            feature_importance.append((sub_feature_idx, average_difference))

        feature_importances[f"Feature Group {eval_index}"] = sorted(feature_importance, key=lambda x: x[1], reverse=True)

    return feature_importances

if __name__ == "__main__":
    # Example usage
    model = MyModel_FinLangNet().to(device)
    model.load_state_dict(torch.load('./FinLangNet.pth'))
    # shape(batch_size, feature_num, feature_length)
    shape_example1 = (256, 20, 200)
    shape_example2 = (256, 170, 220)
    feature1, feature3 = torch.rand(shape_example1)
    feature2 = torch.rand(shape_example2)
    input_features = [feature1, feature3]
    all_inputs = [feature1, feature2, feature3]
    non_evaluated_features_indices = [1] # ignore the second of features

    feature_importances = evaluate_feature_importance(model, input_features, all_inputs, non_evaluated_features_indices)

    feature_groups = {
        'Feature Group 0': "feature1",
        'Feature Group 1': "feature3",
    }

    for group_name, features in feature_importances.items():
        feature_list = feature_groups[group_name]
        print(f"{group_name}:")
        for feature_index, weight in features:
            feature_name = feature_list[feature_index]
            print(f"  Feature: {feature_name}, Index: {feature_index}, Weight: {weight}")
