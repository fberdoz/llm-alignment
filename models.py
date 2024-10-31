import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login

login("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")

SUPPORTED_ENCODER_MODELS = [
    # Sentence-Transformers Models
    'sentence-transformers/all-MiniLM-L6-v2',
    # Developed for fast and efficient sentence embeddings using a small model.
    'sentence-transformers/paraphrase-MiniLM-L6-v2',
    # Created to generate high-quality sentence embeddings for paraphrase identification.
    'sentence-transformers/all-MiniLM-L12-v2',
    # Designed for high-performance sentence embeddings with more layers than L6.
    'sentence-transformers/distiluse-base-multilingual-cased-v2',
    # Multilingual version of USE for generating embeddings across multiple languages.
    'sentence-transformers/use-cmlm-multilingual',
    # Developed for cross-lingual masked language modeling to generate multilingual sentence embeddings.
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    # Created to produce multilingual sentence embeddings for paraphrase detection.
    'sentence-transformers/bert-base-nli-mean-tokens',
    # Designed for natural language inference (NLI) tasks, leveraging BERT and mean pooling.
    'sentence-transformers/bert-base-nli-stsb-mean-tokens',
    # Adapted from BERT to perform sentence similarity tasks using the STS benchmark.
    'sentence-transformers/bert-large-nli-mean-tokens',
    # Larger BERT model trained for NLI with mean pooling for better performance.
    'sentence-transformers/bert-large-nli-stsb-mean-tokens',
    # Larger BERT model fine-tuned for sentence similarity tasks using the STS benchmark.
    'sentence-transformers/roberta-base-nli-mean-tokens',
    # RoBERTa-based model developed for NLI tasks with mean token pooling.
    'sentence-transformers/roberta-large-nli-mean-tokens',
    # Large RoBERTa model trained for NLI tasks with a focus on sentence embeddings.
    'sentence-transformers/distilbert-base-nli-mean-tokens',
    # Lightweight version of BERT for NLI tasks, focusing on efficiency and speed.
    'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
    # A distilled BERT model fine-tuned for sentence similarity using the STS benchmark.
    'sentence-transformers/LaBSE',
    # Language-Agnostic BERT Sentence Embedding model for producing high-quality embeddings across languages.
    'sentence-transformers/msmarco-distilbert-base-v3',
    # Created for passage ranking and information retrieval, fine-tuned on MS MARCO dataset.
    'sentence-transformers/all-mpnet-base-v2',
    # Trained to provide better sentence embeddings using MPNet architecture.
    'sentence-transformers/sentence-t5-base',
    # A smaller version of Sentence-T5, optimized for generating sentence embeddings via a text-to-text framework.
    'sentence-transformers/sentence-t5-large',
    # A larger T5-based model for generating high-quality sentence embeddings using a text-to-text format.

    # HuggingFace Transformers Models
    'bert-base-uncased',
    # Uncased version of BERT, developed for many NLP tasks like question answering and classification.
    'bert-base-cased',  # Cased version of BERT for tasks where case sensitivity is important (e.g., proper nouns).
    'bert-large-uncased',
    # Larger version of BERT for more complex NLP tasks, focusing on deeper context understanding.
    'bert-large-cased',  # Cased version of BERT-large for improved performance on case-sensitive tasks.
    'roberta-base',
    # A robustly optimized version of BERT with modifications, developed to improve pretraining results.
    'roberta-large',  # Larger version of RoBERTa for handling more complex and resource-intensive NLP tasks.
    'distilbert-base-uncased',
    # Lightweight and faster BERT model, distilled for better efficiency with minimal performance loss.
    'distilbert-base-cased',
    # Cased version of DistilBERT for more efficient NLP tasks where case sensitivity is important.
    'gpt2',  # Generative Pretrained Transformer model for text generation, focusing on autoregressive tasks.
    'gpt2-medium',  # Medium-sized GPT-2 for generating longer and more coherent texts compared to smaller models.
    'gpt2-large',
    # Large version of GPT-2 for even higher-quality text generation, but requires more computational resources.
    'gpt2-xl',  # Extra-large GPT-2 model, designed for the most complex text generation tasks requiring large contexts.
    'xlnet-base-cased',
    # Developed to overcome the limitations of BERT by allowing bidirectional and autoregressive training.
    'xlnet-large-cased',
    # Larger version of XLNet for better performance on NLP tasks by leveraging bidirectional context and autoregression.
    'albert-base-v2',
    # A lighter and more efficient version of BERT, developed with parameter-sharing to reduce model size.
    'albert-large-v2',
    # Larger ALBERT model for improved performance on NLP tasks while maintaining efficiency with fewer parameters.
    'albert-xlarge-v2',
    # Extra-large version of ALBERT, designed for more complex NLP tasks requiring deeper understanding.
    'albert-xxlarge-v2',  # Largest ALBERT model for achieving state-of-the-art performance on various NLP benchmarks.
    'google/electra-base-discriminator',
    # Developed for efficient pretraining using a replaced token detection task to improve performance.
    'google/electra-large-discriminator',
    # Larger ELECTRA model for more resource-intensive tasks, designed to improve pretraining efficiency.
    'microsoft/deberta-base',
    # A transformer model with a disentangled attention mechanism for improved performance on various NLP tasks.
    'microsoft/deberta-large',
    # Larger version of DeBERTa, developed for more complex tasks requiring deeper understanding and context.
    'microsoft/deberta-xlarge',  # Extra-large DeBERTa model for state-of-the-art performance on NLP tasks.
    'allenai/longformer-base-4096',
    # Developed to handle long documents with efficient attention mechanisms, extending BERT's capability.
    'allenai/longformer-large-4096',
    # Larger Longformer model for processing long documents with more capacity and better understanding.
    'facebook/bart-base',  # Sequence-to-sequence transformer model for text generation, summarization, and translation.
    'facebook/bart-large',
    # Larger version of BART for handling more complex text generation tasks and larger datasets.
    't5-small',
    # A smaller version of T5 (Text-To-Text Transfer Transformer) for generating text, answering questions, and summarization.
    't5-base',  # Base version of T5 for text generation and understanding tasks in a text-to-text framework.
    't5-large',  # Larger version of T5 for high-performance text generation and understanding tasks.
]


class TextEncoder(nn.Module):
    def __init__(self, model_type):
        super().__init__()
        self.model_type = model_type

        if 'sentence-transformers' in model_type:
            # Sentence-Transformers models, including 'use-cmlm-multilingual'
            self.model = SentenceTransformer(model_type)
            self.output_size = self.model.get_sentence_embedding_dimension()
            self.tokenizer = None  # Handled internally by SentenceTransformer
        else:
            # Other models (BERT, GPT, etc.)
            self.tokenizer = AutoTokenizer.from_pretrained(model_type, clean_up_tokenization_spaces=True)
            self.model = AutoModel.from_pretrained(model_type)
            self.output_size = self.model.config.hidden_size

    def tokenize(self, texts):
        if self.tokenizer:
            return self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        else:
            raise NotImplementedError(f"Tokenization is handled internally for the model type '{self.model_type}'.")

    def forward(self, texts=None):
        if 'sentence-transformers' in self.model_type:
            # Sentence-Transformers handles encoding internally
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            return embeddings
        else:
            tokens = self.tokenize(texts)
            input_ids = tokens['input_ids']
            attention_mask = tokens['attention_mask']
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, 'pooler_output'):
                embeddings = outputs.pooler_output
            else:
                # For models without pooler_output (e.g., GPT)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings


class AgreeDisagreeModel(nn.Module):
    def __init__(self, num_individuals, embedding_dim, encoder_dim, projector_hidden_layers=(), dropout=0, same_weights=True):
        super().__init__()
        self.agree = nn.Embedding(num_individuals, embedding_dim)
        nn.init.xavier_uniform_(self.agree.weight)
        self.disagree = nn.Embedding(num_individuals, embedding_dim)
        nn.init.xavier_uniform_(self.disagree.weight)

        # Copy the weights from 'agree' to 'disagree'
        if same_weights:
            self.disagree.weight.data.copy_(self.agree.weight.data)

        # Create the projector hidden layers (Linear + Relu + Dropout)
        layer_dims = [encoder_dim] + list(projector_hidden_layers)
        linear_layer_dimensions = [(layer_dims[i], layer_dims[i + 1]) for i in range(len(layer_dims) - 1)]
        linear_layers = []
        for in_features, out_features in linear_layer_dimensions:
            linear_layers.append(nn.Linear(in_features, out_features))
            linear_layers.append(nn.ReLU())
            if dropout > 0:
                linear_layers.append(nn.Dropout(dropout))

        # Add the final linear layer to project to the embedding size
        linear_layers.append(nn.Linear(layer_dims[-1], embedding_dim))

        # Assign the projector as a sequential model
        self.projector = nn.Sequential(*linear_layers)

    def forward(self, i, e_q):
        e_a = self.agree(i)
        e_d = self.disagree(i)
        e_q = self.projector(e_q)

        a_score = torch.cosine_similarity(e_q, e_a, dim=-1)
        d_score = torch.cosine_similarity(e_q, e_d, dim=-1)

        #a_score = torch.sum(e_q * e_a, dim=1)
        #d_score = torch.sum(e_q * e_d, dim=1)

        return torch.stack([d_score, a_score], dim=-1)


class AgreeDisagreeLoss(nn.Module):
    def __init__(self, margin=0.0):
        super().__init__()
        self.margin = margin

    def forward(self, scores, labels):
        """
        Args:
            scores (Tensor): A tensor of shape [batch_size, 2], where each row contains two cosine similarities.
            labels (Tensor): A tensor of shape [batch_size], containing labels 0 or 1.

        Returns:
            Tensor: A scalar tensor representing the mean loss over the batch.
        """
        labels = labels.float()
        margin = self.margin

        # Compute loss components based on labels
        # For label == 0:
        #   - We want scores[:, 0] close to 1
        #   - We want scores[:, 1] to be less than or equal to 'margin'
        # For label == 1:
        #   - We want scores[:, 0] to be less than or equal to 'margin'
        #   - We want scores[:, 1] close to 1

        # Calculate losses for both cases
        loss = (
                (1 - labels) * ((1 - scores[:, 0]) + torch.relu(scores[:, 1] - margin)) +
                labels * (torch.relu(scores[:, 0] - margin) + (1 - scores[:, 1]))
        )

        return loss.mean()


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, logits, labels):
        return self.loss(logits, labels.long())