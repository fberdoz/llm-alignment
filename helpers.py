import sys
import numpy as np
from torch.utils.data import Dataset, random_split
import torch
import yaml
import logging
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from models import TextEncoder
import plotly.graph_objects as go

# Custom library from Dustin's project
rsfp_path = '/itet-stor/fberdoz/net_scratch/recommender-systems-for-politics'
sys.path.append(rsfp_path)
from rsfp.data import build_all, build_voters


def write_yaml_experiments(configs, filepath):
    data = {}

    # Create a dictionary with experimentN: { parameter: x_n }
    for i, config in enumerate(configs):
        experiment_key = f"experiment{i}"
        data[experiment_key] = config

    # Write the dictionary to the YAML file (overwrites if it exists)
    with open(filepath, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)


def load_VQA_data(keep_types=("Standard", "Value", "Budget"), normalize=(0, 1), clip_values=False,
                  reset_ids=True, reformulate=True, replace_nan=False, remove_neutral=False, only_full=False):
    logger = logging.getLogger(__name__)

    # Define question types to index map
    q_type = {'Standard': [32214, 32215, 32216, 32217, 32218, 32219, 32220,
                           32221, 32222, 32223, 32224, 32225, 32226, 32227,
                           32228, 32229, 32230, 32231, 32232, 32233, 32234,
                           32235, 32236, 32237, 32238, 32239, 32240, 32241,
                           32242, 32243, 32244, 32245, 32246, 32247, 32248,
                           32249, 32250, 32251, 32252, 32253, 32254, 32255,
                           32256, 32257, 32258, 32259, 32260, 32261, 32262,
                           32263, 32264, 32265, 32266, 32267, 32268, 32269,
                           32270, 32271, 32272, 32273],
              'Value': [32274, 32275, 32276, 32277, 32278, 32279, 32280],
              'Budget': [32281, 32282, 32283, 32284, 32285, 32286, 32287, 32288]}

    # Load data using Dustin's code
    df_v, df_c, df_q = build_all(clean=True, verbose=False)

    # Only keep question ids and questions in english
    df_q = df_q[["ID_question", "question_EN"]].set_index("ID_question")

    # reformulate questions
    if reformulate:
        # Remove the question
        for q_id in q_type["Value"]:
            df_q.at[q_id, 'question_EN'] = df_q.at[q_id, 'question_EN'].split('"')[-2]
        # Remove the "[BePart question" annotation
        for q_id in df_q.index:
            df_q.at[q_id, 'question_EN'] = df_q.at[q_id, 'question_EN'].replace(" [BePart question]", "")
        for q_id in q_type["Budget"]:
            df_q.at[q_id, 'question_EN'] = df_q.at[q_id, 'question_EN'].replace(" or less", "")

    # Set voterID/candidate as index
    df_v.set_index("voterID", inplace=True)
    df_c.set_index("ID_candidate", inplace=True)

    # Only keep answers columns
    columns_to_keep = ['answer_' + str(q_id) for q_id in df_q.index]
    df_v = df_v[columns_to_keep]
    df_c = df_c[columns_to_keep]

    # Rename the answer columns with the question ID as integer
    df_v.rename(columns=lambda col: int(col.split('_')[1]) if col.startswith('answer_') else col, inplace=True)
    df_c.rename(columns=lambda col: int(col.split('_')[1]) if col.startswith('answer_') else col, inplace=True)

    # Remove certain types of questions
    questions_to_keep = []
    for t in keep_types:
        questions_to_keep = questions_to_keep + q_type[t]
    df_v = df_v[questions_to_keep]
    df_c = df_c[questions_to_keep]
    df_q = df_q.loc[questions_to_keep]

    # Drop rows with missing values
    if only_full:
        df_c.dropna(inplace=True)
        df_v.dropna(inplace=True)

    # Clip values to 50 or 100
    if clip_values:
        df_v = df_v.map(lambda x: 100 if x > 50 else (0 if x < 50 else 50))
        df_c = df_c.map(lambda x: 100 if x > 50 else (0 if x < 50 else 50))

    # Remove otherwise
    if remove_neutral:
        df_v = df_v.map(lambda x: None if x == 50 else x)
        df_c = df_c.map(lambda x: None if x == 50 else x)

    # Replace nan
    if replace_nan:
        df_v.fillna(50.0, inplace=True)
        df_c.fillna(50.0, inplace=True)

    # Check for conflicting options
    if replace_nan and remove_neutral:
        logger.warning(
            "'replace_nan' and remove_neutral are both set to True (i.e. neutral answers are set back to their original value).")
    if replace_nan and only_full:
        logger.warning(
            "'replace_nan' and only_full are both set to True (redundant as answers with Nan values are already removed).")

    # Normalize values
    if normalize is not None:
        df_v = normalize[0] + df_v / 100 * (normalize[1] - normalize[0])
        df_c = normalize[0] + df_c / 100 * (normalize[1] - normalize[0])

    # Reset ids to integer numbers
    if reset_ids:
        df_v.reset_index(drop=True, inplace=True)
        df_c.reset_index(drop=True, inplace=True)
        df_q.reset_index(drop=True, inplace=True)
        df_v.columns = range(df_v.shape[1])
        df_c.columns = range(df_c.shape[1])

    # Name the index and columns for readability
    df_v.index.name = "voterID"
    df_v.columns.name = "questionID"
    df_c.index.name = "candidateID"
    df_c.columns.name = "questionID"
    df_q.index.name = "questionID"

    # Print info

    logger.debug("Statistics:")
    logger.debug(f"\tNumber of questions: {len(df_q)}")
    if replace_nan:
        logger.debug(f"\t{len(df_v)} voters with full answers (Nan values were replaced).")
    else:
        logger.debug(
            f"\t{len(df_v)} voters with {df_v.isna().sum().sum()} unanswered questions ({df_v.isna().sum().sum() / df_v.size * 100:.1f}%).")
    logger.debug(f"\t{len(df_c)} candidates (no unanswered question for candidates).")

    return df_v, df_c, df_q


class VQADataset(Dataset):
    def __init__(self, df_answers, question_embeddings=None):
        assert (df_answers.shape[1] == len(question_embeddings))

        self.question_embeddings = question_embeddings
        self.answers = df_answers
        self.idx_map = np.argwhere(~np.isnan(self.answers.values))

        self.n_questions = len(self.answers.columns)
        self.n_individuals = len(self.answers)

    def __len__(self):
        return len(self.idx_map)

    def __getitem__(self, idx):
        a_idx, q_idx = self.idx_map[idx]
        answer = self.answers.iloc[a_idx, q_idx]
        i = torch.tensor(a_idx, dtype=torch.int64)
        a = torch.tensor(answer, dtype=torch.float32)

        if self.question_embeddings is None:
            e_q = None
        else:
            e_q = self.question_embeddings[q_idx]

        q_idx = torch.tensor(q_idx, dtype=torch.int64)
        return i, q_idx, e_q, a


def load_dataset(config, encoder=None):
    logger = logging.getLogger(__name__)
    # Load data
    logger.info("Loading data...")
    df_v, df_c, df_q = load_VQA_data(replace_nan=config['replace_nan'],
                                     clip_values=config['clip_values'],
                                     only_full=config['only_full'],
                                     remove_neutral=config['remove_neutral'])
    logger.info("Data loaded.")

    # Select the data
    if config['answer_type'] == "candidates":
        df = df_c
    elif config['answer_type'] == "voters":
        df = df_v
    else:
        df = df_c
        logger.error(f"Invalid answer type '{config['answer_type']}'. Defaulting to 'candidates'.")

    # Select a subset of individuals
    if 0 < config['num_individuals'] < len(df):
        df = df.sample(config['num_individuals'])

    # Create the embeddings of the questions

    if encoder is not None:
        questions = df_q['question_EN'].tolist()
        with torch.no_grad():
            encoder.eval()
            embeddings = encoder(questions).cpu()

        # Create the dataset
        ds = VQADataset(df, question_embeddings=embeddings)
    else:
        ds = VQADataset(df, question_embeddings=None)

    return ds


def split_dataset(dataset, val_split=0.1, split_mode='random'):
    logger = logging.getLogger(__name__)

    if split_mode == 'questions':
        # TODO: Implement splitting by questions
        raise NotImplementedError("Splitting by questions is not yet implemented.")
    else:
        if split_mode != 'random':
            logger.warning(f"Invalid split mode '{split_mode}'. Defaulting to 'random'.")
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        ds_tr, ds_val = random_split(dataset, [train_size, val_size])

    return ds_tr, ds_val


class MetricTracker:
    def __init__(self):
        """
        scale: The range of the labels. The midpoint will be treated as the 'neutral' answer.
        """

        self.total = 0
        self.metric_buffer = {
            "loss": 0,
            "accuracy": 0,
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0
        }

    def reset_epoch(self):
        """
        Resets the metric  for the next epoch.
        """
        self.total = 0
        self.metric_buffer = {key: 0 for key in self.metric_buffer}

    @staticmethod
    def compute_metrics(outputs, labels, reduction=True):
        logger = logging.getLogger(__name__)

        with torch.no_grad():
            predicted_labels = torch.argmax(outputs, dim=1)
            labels = labels.long()

            if outputs.shape[0] == 0:
                logger.warning("Empty batch. Skipping metrics computation.")
                return {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

            # Confusion matrix components
            TP = torch.sum((predicted_labels == 1) & (labels == 1)).item()
            TN = torch.sum((predicted_labels == 0) & (labels == 0)).item()
            FP = torch.sum((predicted_labels == 1) & (labels == 0)).item()
            FN = torch.sum((predicted_labels == 0) & (labels == 1)).item()

            # Directional accuracy and absolute error
            if reduction:
                acc = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN > 0 else 0
                precision = TP / (TP + FP) if TP + FP > 0 else 0
                recall = TP / (TP + FN) if TP + FN > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
                return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1_score': f1_score}
            else:
                return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}


    def update(self, outputs, labels, precomputed_loss):
        """
        Update the metric buffer with new batch results.

        outputs: Model outputs for the batch.
        labels: Ground truth labels for the batch.
        precomputed_loss: Pre-computed loss for the batch.
        """
        metrics = self.compute_metrics(outputs, labels, reduction=False)
        self.total += outputs.shape[0]

        # Update loss, directional accuracy, and absolute error
        self.metric_buffer["loss"] += precomputed_loss

        # Update TP, TN, FP, and FN
        self.metric_buffer["TP"] += metrics["TP"]
        self.metric_buffer["TN"] += metrics["TN"]
        self.metric_buffer["FP"] += metrics["FP"]
        self.metric_buffer["FN"] += metrics["FN"]

    def get_metrics(self):
        """
        Get the average metrics across all updates so far, including precision, recall, and F1 score.
        """
        avg_metrics = {}

        # Precision, Recall, F1 Score calculations
        TP = self.metric_buffer["TP"]
        TN = self.metric_buffer["TN"]
        FP = self.metric_buffer["FP"]
        FN = self.metric_buffer["FN"]

        loss = self.metric_buffer['loss'] / self.total
        acc = (TP + TN) / self.total if self.total > 0 else 0
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        avg_metrics.update({
            'loss': loss,
            'accuracy': acc,
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        })

        return avg_metrics


class EarlyStopping:
    def __init__(self, patience=10, mode="min"):
        assert mode in ["min", "max"], "Mode must be 'min' or 'max'."

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.mode = mode
        self.stop = False

    def update(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == "min" and score < self.best_score) or (self.mode == "max" and score > self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter > self.patience >= 0:
                self.stop = True


def get_batch_by_value(dataset, value, index=0):
    """
    Returns a batch of samples from the dataset where the element at the specified index equals the given value.

    Args:
    - dataset: A subset of the VQADataset (e.g., ds_tr or ds_val).
    - value: The value to filter the batch by (either i_value or q_value).
    - index: The index of the element to filter by (0 for i, 1 for q).

    Returns:
    - A tuple of tensors (i_batch, q_batch, e_q_batch, a_batch) or None if no matching samples are found.
    """
    logger = logging.getLogger(__name__)
    batch = []

    # Iterate through the dataset to collect all samples where the element at the given index matches the value
    for sample in dataset:
        if sample[index].item() == value:
            batch.append(sample)

    # If there are matching samples, stack them into tensors
    if len(batch) > 0:
        i_batch = torch.stack([item[0] for item in batch])
        q_batch = torch.stack([item[1] for item in batch])
        e_q_batch = [item[2] for item in batch]
        if e_q_batch[0] is not None:
            e_q_batch = torch.stack(e_q_batch)
        else:
            e_q_batch = None
        a_batch = torch.stack([item[3] for item in batch])
        return i_batch, q_batch, e_q_batch, a_batch
    else:
        logger.warning(f"No samples with value = {value} found at index {index}.")
        return None  # If no matching samples found


def evaluate(model, dataset, metric='accuracy', n_individuals=None, n_questions=None):

    # Get list of individuals and questions
    if n_individuals is None:
        individuals = list(range(dataset.dataset.n_individuals))
    else:
        individuals = list(range(min(n_individuals, dataset.dataset.n_individuals)))

    if n_questions is None:
        questions = list(range(dataset.dataset.n_questions))
    else:
        questions = list(range(min(n_questions, dataset.dataset.n_questions)))

    metrics_individuals = {i: None for i in individuals}
    metrics_questions = {q: None for q in questions}


    for individual in individuals:
        batch_i = get_batch_by_value(dataset, individual, index=0)
        if batch_i is None:
            continue
        i, q, e_q, y = batch_i
        with torch.no_grad():
            outputs = model(i, e_q)
        metrics_individuals[individual] = MetricTracker.compute_metrics(outputs, y)[metric]

    for question in questions:
        batch_q = get_batch_by_value(dataset, question, index=1)
        if batch_q is None:
            continue
        i, q, e_q, y = batch_q
        with torch.no_grad():
            outputs = model(i, e_q)
        metrics_questions[question] = MetricTracker.compute_metrics(outputs, y)[metric]

    #    # Filter out None values
    accuracy_individuals = {k: v for k, v in metrics_individuals.items() if v is not None}
    accuracy_questions = {k: v for k, v in metrics_questions.items() if v is not None}

    # Create a subplot with two vertical bars for individuals and questions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

    # Bar plot for individuals
    ax1.bar(accuracy_individuals.keys(), accuracy_individuals.values(), color='blue')
    ax1.set_xlabel('Individual Indexes')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracies of Individuals')

    # Bar plot for questions
    ax2.bar(accuracy_questions.keys(), accuracy_questions.values(), color='green')
    ax2.set_xlabel('Question Indexes')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracies of Questions')

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()


def visualize_scores(model, dataset, value, index=0):
    # Get batch of data for a specific individual/question
    data = get_batch_by_value(dataset, value=value, index=index)
    if data is None:
        print(f"No data found for value = {value} at index = {index}")
        return

    i, q, e_q, y = data

    # Get model outputs
    with torch.no_grad():
        outputs = model(i, e_q)  # Assuming the model outputs two scores per sample

    # Assuming outputs is a tensor of shape [batch_size, 2], where each row contains [score_1, score_2]
    score_1 = outputs[:, 0].cpu().numpy()  # First score
    score_2 = outputs[:, 1].cpu().numpy()  # Second score
    labels = y.cpu().numpy()  # Ground truth labels

    # Visualization
    num_samples = len(score_1)
    x = range(num_samples)  # Sample indices for x-axis

    plt.figure(figsize=(10, 6))

    # Plot model scores (between -1 and 1)
    plt.scatter(x, score_1, label='Disagree', color='red', marker='o')
    plt.scatter(x, score_2, label='Agree', color='green', marker='o')

    # Plot true labels (0 or 1)
    plt.scatter(x, labels, label='True Label', color='black', marker='s')

    # Add small bars between score_1 and score_2
    for idx in range(num_samples):
        if labels[idx] == 0:
            # y == 0: Green if score_1 > score_2, Red if score_1 < score_2
            if score_1[idx] > score_2[idx]:
                plt.vlines(x=idx, ymin=score_2[idx], ymax=score_1[idx], color='green', linewidth=0)
            else:
                plt.vlines(x=idx, ymin=score_1[idx], ymax=score_2[idx], color='red', linewidth=2)
        else:
            # y == 1: Green if score_1 < score_2, Red if score_1 > score_2
            if score_1[idx] < score_2[idx]:
                plt.vlines(x=idx, ymin=score_1[idx], ymax=score_2[idx], color='green', linewidth=0)
            else:
                plt.vlines(x=idx, ymin=score_2[idx], ymax=score_1[idx], color='red', linewidth=2)

    # Add labels and titles
    if index == 0:
        plt.xlabel('Question Index')
    else:
        plt.xlabel('Individual Index')
    plt.ylabel('Scores')
    plt.title('Model Scores and True Labels')
    plt.ylim(-0.1, 1.1)  # Limit y-axis to show both -1 to 1 for scores and 0 or 1 for labels

    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()

def plot_cosine_similarity_heatmap(encoder_type, df_q):
    # Initialize the encoder with the specified model type
    encoder = TextEncoder(encoder_type)

    # Encode all the questions in df_q (assuming df_q contains a 'question_EN' column)
    question_texts = df_q['question_EN'].tolist()
    with torch.no_grad():
        embeddings = encoder(question_texts).cpu().numpy()  # Get embeddings as numpy array

    # Compute cosine similarity between all pairs of embeddings
    cosine_sim_matrix = cosine_similarity(embeddings)

    # Create custom hover text showing both question texts and their cosine similarity
    hover_text = []
    for i in range(len(df_q['question_EN'])):
        hover_row = []
        for j in range(len(df_q['question_EN'])):
            hover_row.append(
                f'Q1: {df_q["question_EN"][i]}<br>Q2: {df_q["question_EN"][j]}<br>Similarity: {cosine_sim_matrix[i, j]:.2f}')
        hover_text.append(hover_row)

    # Create an interactive heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=cosine_sim_matrix,  # Use the full cosine similarity matrix
        x=list(range(len(df_q['question_EN']))),  # Use indices for the x-axis
        y=list(range(len(df_q['question_EN']))),  # Use indices for the y-axis
        colorscale='Viridis',
        colorbar=dict(title='Cosine Similarity'),
        hoverinfo="text",
        text=hover_text,  # Use custom hover text for better interactivity
        hoverongaps=False,
        zmin=-1, zmax=1
    ))

    # Update layout: Remove tick labels from the axes
    fig.update_layout(
        title=f'Cosine Similarity Heatmap of Question Embeddings ({encoder_type})',
        xaxis_title='Questions (index)',
        yaxis_title='Questions (index)',
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        xaxis=dict(showticklabels=False),  # Hide x-axis tick labels
        yaxis=dict(showticklabels=False),  # Hide y-axis tick labels
        height=800, width=800
    )

    # Show the figure
    fig.show()
