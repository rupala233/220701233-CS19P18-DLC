import tensorflow as  
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.models import Sequen al 
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirec onal 
from sklearn.metrics import classifica on_report, confusion_matrix 
import matplotlib.pyplot as plt 
import numpy as np 
 
vocab_size = 10000 
max_length = 200 
 
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size) 
 
x_train = pad_sequences(x_train, maxlen=max_length, padding='post', trunca ng='post') 
x_test = pad_sequences(x_test, maxlen=max_length, padding='post', trunca ng='post') 
 
 
 
model = Sequen al([ 
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length), 
    Bidirec onal(LSTM(64)), 
    Dense(64, ac va on='relu'), 
    Dense(1, ac va on='sigmoid') 
]) 
 
model.compile(loss='binary_crossentropy', opmizer='adam', metrics=['accuracy']) 
model.summary() 
 
 
 
history = model.fit(x_train, y_train, valida on_split=0.2, epochs=5, batch_size=64) 
 
plt.figure(figsize=(12,4)) 
 
plt.subplot(1,2,1) 
plt.plot(history.history['accuracy'], label='Train Accuracy') 
plt.plot(history.history['val_accuracy'], label='Valida on Accuracy') 
plt.legend() 
plt. tle('Accuracy over Epochs') 
 
plt.subplot(1,2,2) 
plt.plot(history.history['loss'], label='Train Loss') 
plt.plot(history.history['val_loss'], label='Valida on Loss') 
plt.legend() 
plt. tle('Loss over Epochs') 
 
plt.show() 
 
 
# Evaluate 
test_loss, test_acc = model.evaluate(x_test, y_test) 
print(f"Test Accuracy: {test_acc:.3f}") 
 
# Predict 
y_pred = (model.predict(x_test) > 0.5).astype(int).fla en() 
 
# Classifica on report 
print(classifica on_report(y_test, y_pred, target_names=['Nega ve', 'Posi ve'])) 
 
# Confusion matrix 
cm = confusion_matrix(y_test, y_pred) 
plt.figure(figsize=(6,6)) 
plt.imshow(cm, cmap='Blues') 
plt. tle('Confusion Matrix') 
plt.colorbar() 
plt.x cks([0,1], ['Nega ve', 'Posi ve']) 
plt.y cks([0,1], ['Nega ve', 'Posi ve']) 
for i in range(2): 
    for j in range(2): 
        plt.text(j, i, cm[i,j], ha='center', va='center', color='red') 
plt.show() 
 
 
# Reverse dic onary for decoding 
word_index = imdb.get_word_index() 
index_word = {v+3:k for k,v in word_index.items()} 
index_word[0] = '<PAD>' 
index_word[1] = '<START>' 
index_word[2] = '<UNK>' 
 
# Func on to decode reviews 
def decode_review(encoded): 
    return ' '.join([index_word.get(i, '?') for i in encoded]) 
 
sample_reviews = [x_test[0], x_test[1]] 
predic ons = model.predict(np.array(sample_reviews)) 
for review, pred in zip(sample_reviews, predic ons): 
    print("Review:", decode_review(review)) 
    print("Predicted Senment:", "Posi ve" if pred>0.5 else "Nega ve") 
    print()
