import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from tqdm import tqdm

# Reads the csv-file into a pandas dataframe
test = pd.read_csv('news_cleaned_2018_02_13.csv')

# Initialize nltk's PorterStemmer
stemmer = PorterStemmer()

# Cleans the data
def clean_content(inp):
    for i in tqdm(range(len(inp))):
        # Converting all content to lower case letters.
        inp = inp.applymap(lambda x:x.lower() if type(x) == str else x)

        # Uses regular expressions to remove and or substitute unwanted substrings with dummy substrings.
        # Removes all newlines and tabs
        inp.at[i,'content'] = re.sub(r"[\n\t]*", "", inp.at[i,'content'])
        # Removes all whitespace that is instantly after a whitespace
        inp.at[i,'content'] = re.sub(r"[\s]{2,}", "", inp.at[i,'content'])
        # Sub of dates
        inp.at[i,'content'] = re.sub(r"(([a-zA-Z]*)(\s+)(\d{2,})(,{1})(\s+)(\d{2,4}))","uniquedate", inp.at[i,'content'], flags=re.MULTILINE)
        # Sub of emails
        inp.at[i,'content'] = re.sub(r"([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)", "uniqueemail", inp.at[i,'content'], flags=re.MULTILINE)
        # Sub of url's
        inp.at[i,'content'] = re.sub(r"(?:https?:\/\/)?(?:www\.)?([^@\s]+\.[a-zA-Z]{2,4})[^\s]*","uniqueurl", inp.at[i,'content'], flags=re.MULTILINE)
        #Removal of numbers - (\s)\$?(?:[\d,.-])+
        inp.at[i,'content'] = re.sub(r"\b(\d+)\b","uniquenum", inp.at[i,'content'], flags=re.MULTILINE)
    return inp


data = next(test)

# Clean the data
data = clean_content(data)

# Save the cleaned data to a new CSV file
data.to_csv('news_cleaned.csv', index=False)


import pandas as pd
import nltk 

# read in the data and sample 10% of the rows
data = pd.read_csv('news_cleaned.csv').sample(frac=0.1)

# Tokenize the text
data['tokens'] = data['content'].apply(nltk.word_tokenize)

# Remove stopwords and compute the size of the vocabulary
stop_words = set(stopwords.words('english'))
data['tokens'] = data['tokens'].apply(lambda x: [word for word in x if word.lower() not in stop_words])
vocab_size = len(set([word for row in data['tokens'] for word in row]))
reduction_rate_stopwords = (1 - (vocab_size / len(set([word for row in data['content'] for word in row.split()])))) * 100

# Remove word variations with stemming and compute the size of the vocabulary
ps = PorterStemmer()
data['tokens'] = data['tokens'].apply(lambda x: [ps.stem(word) for word in x])
vocab_size_stem = len(set([word for row in data['tokens'] for word in row]))
reduction_rate_stemming = (1 - (vocab_size_stem / vocab_size)) * 100

print(f"Vocabulary size before removing stopwords: {len(set([word for row in data['content'] for word in row.split()]))}")
print(f"Vocabulary size after removing stopwords: {vocab_size}")
print(f"Reduction rate after removing stopwords: {reduction_rate_stopwords:.2f}%")
print(f"Vocabulary size after stemming: {vocab_size_stem}")
print(f"Reduction rate after stemming: {reduction_rate_stemming:.2f}%")




import numpy as np

# Define the split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Split the data into training, validation, and test sets
num_rows = len(data)
indices = np.arange(num_rows)
np.random.shuffle(indices)
train_idx = indices[:int(train_ratio*num_rows)]
val_idx = indices[int(train_ratio*num_rows):int((train_ratio+val_ratio)*num_rows)]
test_idx = indices[int((train_ratio+val_ratio)*num_rows):]

train_data = data.iloc[train_idx]
val_data = data.iloc[val_idx]
test_data = data.iloc[test_idx]



import random

def random_baseline(data):
    data['predicted'] = [random.choice(['reliable', 'fake']) for _ in range(len(data))]
    return data

def calculate_accuracy(data, true_labels):
    correct_predictions = data[data['predicted'] == true_labels]
    accuracy = len(correct_predictions) / len(data)
    return accuracy

# Example usage:
random_data = random_baseline(test_data)
accuracy = calculate_accuracy(random_data, test_data['type'])
print("Random Baseline Accuracy:", accuracy)


from sklearn.metrics import classification_report

# Convert target labels to integers
target_labels = test_data['type'].apply(lambda x: 0 if x == 'reliable' else 1)

# Convert predicted labels to integers
predicted_labels = random_data['predicted'].apply(lambda x: 0 if x == 'reliable' else 1)

# Compute classification report
report = classification_report(target_labels, predicted_labels)

# Print classification report
print(report)



import pandas as pd
import random

# Load data
data = pd.read_csv("news_cleaned.csv")

# Define train, validation, and test sets
train_size = int(len(data) * 0.6)
val_size = int(len(data) * 0.2)
test_size = len(data) - train_size - val_size

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# Define Most Common Label model
def MCL(train_data, val_data):
    # Get most common label from training set
    mcl = train_data['type'].value_counts().idxmax()

    # Make predictions for training and validation sets
    train_data['predicted'] = mcl
    val_data['predicted'] = mcl

    # Calculate accuracy
    train_accuracy = (train_data['type'] == train_data['predicted']).mean()
    val_accuracy = (val_data['type'] == val_data['predicted']).mean()

    return train_accuracy, val_accuracy

# Run MCL model on data
train_accuracy, val_accuracy = MCL(train_data, val_data)

# Print results
print("Most Common Label model:")
print(f"Training accuracy: {train_accuracy:.3f}")
print(f"Validation accuracy: {val_accuracy:.3f}")

def RandomForest(training_data, validation_data):
    # vectorize the training data
    vectorizer = CountVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
    X_train_vect = vectorizer.fit_transform(training_data['content'].values.astype('U'))
    y_train = training_data['content'].values
    
    # vectorize the validation data
    X_val_vect = vectorizer.transform(validation_data['content'].values.astype('U'))
    y_val = validation_data['content'].values

    # initialize and fit the random forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_vect.toarray(), y_train)
    
    # make predictions and calculate accuracy
    y_pred = rf.predict(X_val_vect.toarray())
    accuracy = (y_pred == y_val).mean()

    return {'model': rf, 'vectorizer': vectorizer, 'accuracy': accuracy}



# split the data into non-null rows
training_data = train_data.dropna(subset=['content'])
validation_data = val_data.dropna(subset=['content'])

# check that we have at least one non-null row in each set
if training_data.shape[0] == 0:
    raise ValueError("Training data is empty!")
if validation_data.shape[0] == 0:
    print("Warning: Validation data is empty. Setting predictions to empty list.")
    data = []
else:
    # run the model
    data = RandomForest(training_data, validation_data)
    
print(data)



text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),])
text_clf = text_clf.fit(df['content'], df['label'])

def Model3(train, validate): 
    
    #X_train_counts, X_train_tfidf = PrepareData(train)
    
    text_clf_svm = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=100, random_state=42)),
     ])

    _ = text_clf_svm.fit(train['content'], train['label'])
     
    # Make Predictions on train data
    train_predicted = text_clf_svm.predict(train['content'])
    
    # Make Predictions on validation data
    predicted = text_clf.predict(validate['content'])
    
    #np.mean(predicted == validation['label'])

    train['predicted'] = train_predicted
    validate['predicted'] = predicted
    
    return train[['label','predicted']], validate[['label','predicted']]



def PrepareData(train):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train['content'])

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    #print("TF-IDF: Document-Term matrix shape:", X_train_tfidf.shape)
    return X_train_counts, X_train_tfidf


def Model4(train, validate): 
    
    X_train_counts, X_train_tfidf = PrepareData(train)
    
    # Naive Bayes    
    clf = MultinomialNB().fit(X_train_tfidf, train['label'])
    
    # Building a pipeline: We can write less code and do all of the above, by building a pipeline as follows:
    # The names ‘vect’ , ‘tfidf’ and ‘clf’ are arbitrary but will be used later.
    text_clf = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', MultinomialNB()),])
    
    # Train the Naive Bayes Classifier on train data
    text_clf = text_clf.fit(train['content'], train['label'])
    
    # Make Predictions on train data
    train_predicted = text_clf.predict(train['content'])
    
    # Make Predictions on validation data
    predicted = text_clf.predict(validate['content'])
    
    #np.mean(predicted == validation['label'])

    train['predicted'] = train_predicted
    validate['predicted'] = predicted
    
    return train[['label','predicted']], validate[['label','predicted']]


x = Model4(data[['content','label']],data[['content','label']])
print(x)




def decode_article(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
    
def Model5(train,validate):
    vocab_size = 5000
    embedding_dim = 64
    max_length = 200
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok =  '<OOV>'

    train_articles = train['content']
    train_labels = train['label']
    
    validation_articles = validate['content']
    validation_labels = validate['label']
        
    #
    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_articles)
    word_index = tokenizer.word_index
    dict(list(word_index.items())[0:10])
    print("word_index",word_index)
    
    
    train_sequences = tokenizer.texts_to_sequences(train_articles)
    print(train_sequences[10])

    # 
    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    train_padded = pad_sequences(train_sequences,padding=padding_type, truncating=trunc_type)
    print(train_padded[10])

    #
    validation_sequences = tokenizer.texts_to_sequences(validation_articles)
    validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)    
    print("validation_sequences", len(validation_sequences))
    print("validation_padded", validation_padded.shape)
    
    #
    print("train_labels", train_labels)
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(word_index)
    
    training_label_seq = np.array(train_labels)
    validation_label_seq = np.array(validation_labels)
    training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
    validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    model5 = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')])  
        
    model5.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    num_epochs = 5
    history = model5.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)
    save_model(model5)
    with open('model_pkl', 'wb') as files:
        pickle.dump(model5, files)      
       
    print(history.params)
        
    # check the keys of history object
    print(history.history.keys())

    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_'+string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_'+string])
        plt.show()
    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')    
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    def decode_article(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])
    print(decode_article(train_padded[10]))
    print(train_articles[10])
    # Get predictions
    x = model5.predict(train_padded) 
    x_class = x.argmax(axis=-1)  
    
    # Save numpy arrays to df
    x_df = pd.DataFrame(train_labels,columns=['label'])      
    x_class_df = pd.DataFrame(x_class,columns=['predicted'])  
    print(x_df)
    print(x_class_df)
    # Combine df's
    x_df_combined = pd.concat([x_df['label'].astype(int).reset_index(),x_class_df['predicted'].astype(int).reset_index()],axis=1) 
    print("x_df_combined",x_df_combined)
    # Get predictions
    y = model5.predict(validation_padded) 
    y_class = y.argmax(axis=-1)
    # Save numpy arrays to df
    y_df = pd.DataFrame(validation_labels,columns=['label'])    
    y_class_df = pd.DataFrame(y_class,columns=['predicted'])
    # Combine df's
    y_df_combined = pd.concat([y_df['label'].astype(int).reset_index(),y_class_df['predicted'].astype(int).reset_index()],axis=1) 
    #print("validation_labels_df_combined",y_df_combined)
    return x_df_combined[['label','predicted']], y_df_combined[['label','predicted']]

return_5 = Model5(df.loc[0:9], df.loc[10:19])
