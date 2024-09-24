# models.py

from typing import List

import torch
import torch.nn as nn
from torch import optim

from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """

    def __init__(self, embedding_layer, num_classes, hidden_dim=50):
        super(NeuralSentimentClassifier, self).__init__()
        self.embedding_layer = embedding_layer
        self.fc1 = nn.Linear(self.embedding_layer.embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding_layer(x)  # Embedding the input indices
        x = torch.mean(x, dim=1)  # Averaging the embeddings
        x = torch.relu(self.fc1(x))  # First fully connected layer with ReLU activation
        x = self.fc2(x)  # Output layer
        return torch.log_softmax(x, dim=1)

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        # Convert words to indices and run prediction
        indices = [self.word_indexer.get_index(word) for word in ex_words]
        indices_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            log_probs = self.forward(indices_tensor)
        return torch.argmax(log_probs, dim=1).item()


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings,
                                 train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
    # Prepare the data
    vocab_size = len(word_embeddings.word_indexer)
    embedding_dim = word_embeddings.vectors.shape[1]
    embedding_layer = nn.Embedding.from_pretrained(torch.tensor(word_embeddings.vectors, dtype=torch.float),
                                                   freeze=False)

    model = NeuralSentimentClassifier(embedding_layer, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for example in train_exs:
            optimizer.zero_grad()
            input_ids = torch.tensor([word_embeddings.word_indexer.get_index(word) for word in example.words],
                                     dtype=torch.long).unsqueeze(0)
            labels = torch.tensor([example.label], dtype=torch.long)
            output = model(input_ids)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_exs)}')

        # Optional: Evaluate on dev set
    return model
