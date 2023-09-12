import numpy as np
import sys
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding, Flatten
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
import twitter_data_handler

# Performs classification using CNN.

FREQ_DIST_FILE = 'dataset/data-processed-freqdist.pkl'
BI_FREQ_DIST_FILE = 'dataset/data-processed-freqdist-bi.pkl'
TRAIN_PROCESSED_FILE = 'dataset/data-processed.csv'
TEST_PROCESSED_FILE = 'dataset/test-processed.csv'
GLOVE_FILE = 'dataset/glove-seeds.txt'
dim = 50
vocab_size = 90000
vocab = utils.top_n_words(FREQ_DIST_FILE, vocab_size, shift=1)


def get_glove_vectors(vocab):
    """
    Extracts glove vectors from seed file only for words present in vocab.
    """
    print('Looking for GLOVE seeds')
    glove_vectors = {}
    found = 0
    with open(GLOVE_FILE, 'r', encoding="utf8") as glove_file:
        for i, line in enumerate(glove_file):
            utils.write_status(i + 1, 0)
            tokens = line.strip().split()
            word = tokens[0]
            if vocab.get(word):
                vector = [float(e) for e in tokens[1:]]
                glove_vectors[word] = np.array(vector)
                found += 1
    print('\n')
    return glove_vectors


def get_feature_vector(tweet):
    """
    Generates a feature vector for each tweet where each word is
    represented by integer index based on rank in vocabulary.
    """
    words = tweet.split()
    feature_vector = []
    for i in range(len(words) - 1):
        word = words[i]
        if vocab.get(word) is not None:
            feature_vector.append(vocab.get(word))
    if len(words) >= 1:
        if vocab.get(words[-1]) is not None:
            feature_vector.append(vocab.get(words[-1]))
    return feature_vector


def process_tweets(csv_file, test_file=True):
    """
    Generates training X, y pairs.
    """
    tweets = []
    labels = []
    print('Generating feature vectors')
    with open(csv_file, 'r') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            if test_file:
                tweet_id, tweet = line.split(',')
            else:
                tweet_id, sentiment, tweet = line.split(',')
            feature_vector = get_feature_vector(tweet)
            if test_file:
                tweets.append(feature_vector)
            else:
                tweets.append(feature_vector)
                labels.append(int(sentiment))
            utils.write_status(i + 1, total)
    print('\n')
    return tweets, np.array(labels)


def get_tweets_from_feed():
    tweets_array = twitter_data_handler.get_tweets()
    tweets = []
    labels = []
    for i in range(len(tweets_array)):
        # id = tweets_array[i].id
        text = tweets_array[i].text
        print('Generating feature vectors >> ' + text)
        feature_vector = get_feature_vector(text)
        tweets.append(feature_vector)
        utils.write_status(i + 1, len(tweets_array))
        print('\n')
    return tweets, np.array(labels)


def run_model(train):
    np.random.seed(1337)
    batch_size = 500
    max_length = 40
    filters = 600
    kernel_size = 3
    glove_vectors = get_glove_vectors(vocab)
    tweets, labels = process_tweets(TRAIN_PROCESSED_FILE, test_file=False)
    # Create and embedding matrix
    embedding_matrix = np.random.randn(vocab_size + 1, dim) * 0.01
    # Seed it with GloVe vectors

    for word, i in vocab.items():
        glove_vector = glove_vectors.get(word)
        if glove_vector is not None:
            embedding_matrix[i] = glove_vector
    tweets = pad_sequences(tweets, maxlen=max_length, padding='post')
    shuffled_indices = np.random.permutation(tweets.shape[0])
    tweets = tweets[shuffled_indices]
    labels = labels[shuffled_indices]
    if train:
        model = Sequential()
        model.add(Embedding(vocab_size + 1, dim, weights=[embedding_matrix], input_length=max_length))
        model.add(Dropout(0.4))
        model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
        model.add(Conv1D(300, kernel_size, padding='valid', activation='relu', strides=1))
        model.add(Conv1D(150, kernel_size, padding='valid', activation='relu', strides=1))
        model.add(Conv1D(75, kernel_size, padding='valid', activation='relu', strides=1))
        model.add(Flatten())
        model.add(Dense(600))
        model.add(Dropout(0.5))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        filepath = "models/4cnn-{epoch:02d}-{loss:0.3f}-{accuracy:0.3f}-{val_loss:0.3f}-{val_accuracy:0.3f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1, save_best_only=True, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001)
        model.fit(tweets, labels, batch_size=128, epochs=8, validation_split=0.1, shuffle=True,
                  callbacks=[checkpoint, reduce_lr])
    else:
        model = load_model('models/4cnn-08-0.341-0.849-0.498-0.768.hdf5')
        print(model.summary())
        test_tweets, _ = get_tweets_from_feed()
        test_tweets = pad_sequences(test_tweets, maxlen=max_length, padding='post')
        predictions = model.predict(test_tweets, batch_size=128, verbose=1)
        print(predictions)
        results = zip(map(str, range(len(test_tweets))), np.round(predictions[:, 0]).astype(int))
        utils.save_results_to_csv(results, '../results/tweets_predictions.csv')


if __name__ == '__main__':
    run_model(True)
