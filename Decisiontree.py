import numpy as np

def calculate_entropy(n_category1: int, n_category2: int) -> float:
    """
    Calculate the entropy of a binary classification.

    Parameters:
    - n_category1 (int): Number of samples in category 1.
    - n_category2 (int): Number of samples in category 2.

    Returns:
    - float: The calculated entropy.
    """
    total = n_category1 + n_category2
    probabilities = [n_category1 / total, n_category2 / total]
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    return entropy

def calculate_information_gain(pre_split_x, pre_split_o, post_split_x, post_split_o):
    """
    Calculate the information gain from a binary split.

    Parameters:
    - pre_split_x (int): Number of samples in category X before the split.
    - pre_split_o (int): Number of samples in category O before the split.
    - post_split_x (int): Number of samples in category X after the split.
    - post_split_o (int): Number of samples in category O after the split.

    Returns:
    - float: The calculated information gain.
    """
    total_pre_split = pre_split_x + pre_split_o
    total_post_split = post_split_x + post_split_o
    total = total_pre_split + total_post_split
    entropy_before = calculate_entropy(pre_split_x + post_split_x, pre_split_o + post_split_o)
    entropy_after = (total_pre_split / total * calculate_entropy(pre_split_x, pre_split_o) + 
                     total_post_split / total * calculate_entropy(post_split_x, post_split_o))
    information_gain = entropy_before - entropy_after
    return information_gain

def find_optimal_split(features: np.ndarray, labels: np.ndarray) -> (int, float):
    """
    Find the feature and value for the optimal binary split based on information gain.

    Parameters:
    - features (np.ndarray): The feature matrix.
    - labels (np.ndarray): The labels vector.

    Returns:
    - int: The column index of the optimal feature.
    - float: The value for the optimal split.
    """
    best_index = best_value = best_gain = 0
    for column in range(features.shape[1]):
        unique_values = np.unique(features[:, column])
        for value in unique_values:
            mask = features[:, column] <= value
            gain = calculate_information_gain(
                sum(mask & (labels == 0)), sum(mask & (labels == 1)),
                sum(~mask & (labels == 0)), sum(~mask & (labels == 1))
            )
            if gain > best_gain:
                best_gain, best_index, best_value = gain, column, value
    return best_index, best_value

class DecisionTreeNode:
    """
    A node in the decision tree.

    Attributes:
    - decision (Optional[int]): The class decision at this node (for leaf nodes).
    - feature (Optional[int]): The feature index used for splitting.
    - value (Optional[float]): The value used for splitting.
    - left (Optional[DecisionTreeNode]): The left child node.
    - right (Optional[DecisionTreeNode]): The right child node.
    """
    def __init__(self, decision=None, feature=None, value=None, left=None, right=None):
        self.decision = decision
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right

    def predict(self, features):
        """
        Predict the class label for a given feature set.

        Parameters:
        - features (np.ndarray): The feature set to predict.

        Returns:
        - int: The predicted class label.
        """
        if self.decision is not None:
            return self.decision
        elif features[self.feature] <= self.value:
            return self.left.predict(features)
        else:
            return self.right.predict(features)

def build_decision_tree(features: np.ndarray, labels: np.ndarray) -> DecisionTreeNode:
    """
    Build a decision tree using the given features and labels.

    Parameters:
    - features (np.ndarray): The feature matrix.
    - labels (np.ndarray): The labels vector.

    Returns:
    - DecisionTreeNode: The root node of the constructed decision tree.
    """
    if np.unique(labels).size == 1:
        return DecisionTreeNode(decision=labels[0])
    best_feature, best_value = find_optimal_split(features, labels)
    mask = features[:, best_feature] <= best_value
    left_subtree = build_decision_tree(features[mask], labels[mask])
    right_subtree = build_decision_tree(features[~mask], labels[~mask])
    return DecisionTreeNode(feature=best_feature, value=best_value, left=left_subtree, right=right_subtree)

def read_data(filename: str):
    """
    Read feature vectors and labels from a file.

    Parameters:
    - filename (str): The path to the data file.

    Returns:
    - tuple: A tuple containing the feature matrix and labels vector.
    """
    features, labels = [], []
    try:
        with open(filename, 'r') as file:
            for line in file:
                elements = [int(value) for value in line.strip().split(',')]
                features.append(elements[:-1])  # All but the last element
                labels.append(elements[-1])     # The last element
    except FileNotFoundError:
        print(f"File {filename} not found.")
        raise

    return np.array(features), np.array(labels)

def read_test_data(filename: str):
    """
    Read test feature vectors from a file.

    Parameters:
    - filename (str): The path to the test data file.

    Returns:
    - np.ndarray: The feature matrix for the test data.
    """
    features = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                elements = [int(value) for value in line.strip().split(',')]
                features.append(elements)
    except FileNotFoundError:
        print(f"File {filename} not found.")
        raise

    return np.array(features)

def write_predictions_to_file(predictions: np.ndarray, output_filename: str):
    """
    Write the predicted labels to a file.

    Parameters:
    - predictions (np.ndarray): The predicted labels.
    - output_filename (str): The path to the output file.
    """
    try:
        with open(output_filename, 'w') as file:
            for prediction in predictions:
                file.write(f"{prediction}\n")
    except Exception as e:
        print(f"Failed to write to {output_filename}: {e}")
        raise

def predict_labels(tree: DecisionTreeNode, features: np.ndarray):
    """
    Predict labels for a given set of features using the decision tree.

    Parameters:
    - tree (DecisionTreeNode): The decision tree for prediction.
    - features (np.ndarray): The feature matrix.

    Returns:
    - np.ndarray: The predicted labels.
    """
    return np.array([tree.predict(feature) for feature in features])

def main():
    """
    Main function to execute the decision tree algorithm.
    """
    train_file = 'train.csv'
    test_file = 'test.csv'
    output_file = 'results.csv'

    # Load training and test data
    train_features, train_labels = read_data(train_file)
    test_features = read_test_data(test_file)

    # Build the decision tree
    decision_tree = build_decision_tree(train_features, train_labels)

    # Predict and write test labels to file
    test_predictions = predict_labels(decision_tree, test_features)
    write_predictions_to_file(test_predictions, output_file)

    print(f"Decision tree model predictions written to {output_file}")

if __name__ == "__main__":
    main()

