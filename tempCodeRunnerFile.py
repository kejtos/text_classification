import csv
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_score
from sklearn.utils import check_random_state

from scipy.sparse import coo_matrix

import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

file = 'attributes.tsv'
file_cleared = 'attributes_cleared.tsv'
file_clustered = 'attributes_clustered.tsv'

random_state = check_random_state(11)
np.random.seed(11)
tf.random.set_seed(11)
csv.field_size_limit(10000000)

## Reading and preprocessing the data
# Removing HTML tags and empty rows
data = {}
with open(file, 'r', encoding='utf-8') as rf, open(file_cleared, 'w+', encoding='utf-8', newline='') as wf:
    reader = csv.reader(rf, delimiter='\t')
    writer = csv.writer(wf, delimiter='\t')

    for row in reader:
        soup = BeautifulSoup(row[1], 'html.parser')
        row[1] = soup.get_text()

        if row[1].strip():
            _ = writer.writerow(row)
            data.setdefault(row[0], []).append(row[1])

data = {id: ' '.join(texts) for id, texts in data.items()}
_ = data.pop('Object_Id')

# Vectorizing the data
vectorizer = TfidfVectorizer()
vecs = vectorizer.fit_transform(data.values())
coo_vecs = coo_matrix(vecs)

# Convertïng to SparseTensor
sparse_tensor = tf.SparseTensor(indices=np.transpose([coo_vecs.row, coo_vecs.col]), values=coo_vecs.data, dense_shape=coo_vecs.shape)
reordered_vecs = tf.sparse.reorder(sparse_tensor)
dense_vecs = tf.sparse.to_dense(reordered_vecs)

# Normalizing
norm_vecs = tf.keras.utils.normalize(dense_vecs, axis=1)
input_dim = norm_vecs.shape[1]

# Defining encoding dimensions
encoding_dim = 2**np.arange(2, 7)
input_data = Input(shape=(input_dim,))
sil_birch = []

labs = np.array([])

# Grid search for the best encoding dimensions and threshold for Birch
for i, _ in np.ndenumerate(encoding_dim):
    for t in range(1, 10):
        # Training the autoencoder
        encoded = Dense(encoding_dim[i], activation='relu')(input_data)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)

        autoencoder = Model(input_data, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
        autoencoder.fit(norm_vecs, norm_vecs, epochs=10, batch_size=32)

        # Extract the encoder model and encode the data
        encoder = Model(input_data, encoded)
        encoded_data = encoder.predict(norm_vecs)

        # Cluster
        birch = Birch(threshold=t/10)
        _ = birch.fit(encoded_data)

        birch_labels = birch.labels_
        max_label = np.max(birch_labels)

        silhouette = silhouette_score(encoded_data, birch_labels)
        sil_birch.append((encoding_dim[i], t/10, silhouette, birch_labels))

# Create a DataFrame from the silhouette scores
sils = pd.DataFrame(sil_birch)
sils.columns = ['Enconding dimensions', 'Threshold', 'Silhouette Score', 'Cluster_Id']

# Find the row with the maximum silhouette score
id_max = sils['Silhouette Score'].idxmax()
best_result = sils.loc[id_max,:]

results = pd.DataFrame(dict(Object_Id=data.keys(), Cluster_Id=sils.loc[id_max, 'Cluster_Id']))

results.to_csv(file_clustered, sep='\t')
