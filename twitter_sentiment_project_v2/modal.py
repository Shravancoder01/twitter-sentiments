import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud  
from collections import Counter  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier
)

import os
import pickle

# ------------------- DO NOT REMOVE: Required NLTK downloads -------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

# ------------------- DATA LOADING -------------------
csv_path = os.path.join("data", "Twitter_Data.csv")
raw = pd.read_csv(csv_path, encoding="utf-8")

# Use dataset’s real columns: clean_text, category
df = raw[['category', 'clean_text']].rename(columns={'category': 'Target', 'clean_text': 'Text'})

# ------------------- Normalize Labels -------------------
# Convert -1/0/1 into friendly names
_target_name_map = {-1: 'negative', 0: 'neutral', 1: 'positive'}
df['TargetName'] = df['Target'].map(_target_name_map)

encoder = LabelEncoder()
df['Target'] = encoder.fit_transform(df['TargetName'])  # maps negative→0, neutral→1, positive→2
print(df.head(5))

# ------------------- Drop unused columns if present -------------------
for col in ['Id', 'Date', 'Flag', 'User']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# ------------------- RENAME STEP (no-op here) -------------------
print("Rename Columns")
df.rename(columns={'v1': 'Target', 'v2': 'Text'}, inplace=True, errors='ignore')  # avoid error if not present
print(df.sample(min(5, len(df))))

# ------------------- Encode Labels (already encoded above) -------------------
print("Refilling the Data in Column = Target")
print(df['Target'].unique())

# ------------------- Nulls & Duplicates -------------------
print("\nCheck for null values or missing values")
print(df.isnull().sum())
df.dropna(subset=['Text', 'Target'], inplace=True)

print("\nCheck for duplicate values")
print(df.duplicated().sum())

print("\nRemove duplicate values")
df = df.drop_duplicates(keep='first')
print(df.head())

# ------------------- Target distribution & shape -------------------
print("\nReading Target Data")
print(df['Target'].value_counts())

print("\nShape of Dataset")
print(df['Target'].shape)
print(df.shape)

# ------------------- Pie Chart -------------------
plt.pie(df['Target'].value_counts(), labels=encoder.classes_, autopct='%1.1f%%')
plt.legend()
plt.title("Pie Chart of Target Variable")
plt.axis('equal')
plt.show()

# ------------------- Feature Engineering -------------------
df['num_characters'] = df['Text'].astype(str).apply(len)
print(df['num_characters'])

df['word_list'] = df['Text'].astype(str).apply(lambda x: nltk.word_tokenize(x))
print(df.head())

df['word_count'] = df['word_list'].apply(len)
print(df.head())

df['Sent_list'] = df['Text'].astype(str).apply(lambda x: nltk.sent_tokenize(x))
df['Sent_count'] = df['Sent_list'].apply(len)
print(df.head())

print(df[['Text', 'num_characters', 'word_list', 'word_count', 'Sent_list', 'Sent_count']].describe())

# ------------------- Positive/Negative/Neutral Analysis -------------------
cls_to_id = {c: i for i, c in enumerate(encoder.classes_)}
pos_id = cls_to_id.get('positive')
neg_id = cls_to_id.get('negative')
neu_id = cls_to_id.get('neutral')

if pos_id is not None:
    print("\nAnalysis for positive Data")
    print(df[df['Target'] == pos_id][['Text', 'num_characters', 'word_count', 'Sent_count']].describe())

if neg_id is not None:
    print("\nAnalysis for negative Data")
    print(df[df['Target'] == neg_id][['Text', 'num_characters', 'word_count', 'Sent_count']].describe())

if neu_id is not None:
    print("\nAnalysis for neutral Data")
    print(df[df['Target'] == neu_id][['Text', 'num_characters', 'word_count', 'Sent_count']].describe())

# ------------------- Visualization -------------------
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.histplot(df[df['Target'] == pos_id], x='num_characters', bins=50, label='Positive', kde=True, ax=axes[0])
sns.histplot(df[df['Target'] == neg_id], x='num_characters', bins=50, label='Negative', kde=True, ax=axes[0])
if neu_id is not None:
    sns.histplot(df[df['Target'] == neu_id], x='num_characters', bins=50, label='Neutral', kde=True, ax=axes[0])
axes[0].set_title('Characters Count Distribution')
axes[0].set_xlabel('Number of Characters')
axes[0].set_ylabel('Frequency')
axes[0].legend()

sns.histplot(df[df['Target'] == pos_id], x='word_count', bins=50, label='Positive', kde=True, ax=axes[1])
sns.histplot(df[df['Target'] == neg_id], x='word_count', bins=50, label='Negative', kde=True, ax=axes[1])
if neu_id is not None:
    sns.histplot(df[df['Target'] == neu_id], x='word_count', bins=50, label='Neutral', kde=True, ax=axes[1])
axes[1].set_title('Word Count Distribution')
axes[1].set_xlabel('Number of Words')
axes[1].set_ylabel('Frequency')
axes[1].legend()

sns.histplot(df[df['Target'] == pos_id], x='Sent_count', bins=50, label='Positive', kde=True, ax=axes[2])
sns.histplot(df[df['Target'] == neg_id], x='Sent_count', bins=50, label='Negative', kde=True, ax=axes[2])
if neu_id is not None:
    sns.histplot(df[df['Target'] == neu_id], x='Sent_count', bins=50, label='Neutral', kde=True, ax=axes[2])
axes[2].set_title('Sentence Count Distribution')
axes[2].set_xlabel('Number of Sentences')
axes[2].set_ylabel('Frequency')
axes[2].legend()

plt.tight_layout()
plt.show()

try:
    sample_df = df.sample(500) if len(df) > 1000 else df
    sns.pairplot(sample_df[['Target', 'num_characters', 'word_count', 'Sent_count']], hue='Target')
    plt.show()
except Exception as e:
    print("Pairplot skipped:", e)

numeric_df = df[['Target', 'num_characters', 'word_count', 'Sent_count']]
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='seismic')
plt.title("Correlation Heatmap {numeric features}")
plt.show()

# ------------------- data processing -------------------
def transform_text(text):
    text = str(text).lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english'):
            y.append(i)

    text = y[:]
    ps = PorterStemmer()
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

print()
output = transform_text("Hi Hello how are you @ ? !/Raj DRives a car . Mary loves eating pizza,")
print(f"Final Output : {output}")

df['Transformed_Text'] = df['Text'].apply(transform_text)
print(df.describe())
print(df.info())

# ------------------- Word clouds -------------------
def corpus_for(tid):
    return df[df['Target'] == tid]['Transformed_Text'].str.cat(sep=" ")

cls_to_id = {c: i for i, c in enumerate(encoder.classes_)}
neg_id = cls_to_id.get('negative')
neu_id = cls_to_id.get('neutral')
pos_id = cls_to_id.get('positive')

neg_corpus = corpus_for(neg_id) if neg_id is not None else ""
neu_corpus = corpus_for(neu_id) if neu_id is not None else ""
pos_corpus = corpus_for(pos_id) if pos_id is not None else ""

neg_wc = None
pos_wc = None
neu_wc = None

if len(neg_corpus):
    neg_wc = WordCloud(width=600, height=600, background_color='black', max_words=200, colormap='Reds').generate(neg_corpus)
if len(pos_corpus):
    pos_wc = WordCloud(width=600, height=600, background_color='white', max_words=200, colormap='Blues').generate(pos_corpus)
if len(neu_corpus):
    neu_wc = WordCloud(width=600, height=600, background_color='white', max_words=200, colormap='Greens').generate(neu_corpus)

# ------------------- Top words -------------------
def top_words_for(tid):
    corpus = []
    for msg in df[df['Target'] == tid]['Transformed_Text'].tolist():
        for words in msg.split():
            corpus.append(words)
    return Counter(corpus).most_common(30)

if neg_id is not None:
    top_30_negative = top_words_for(neg_id)
    df_top_negative = pd.DataFrame(top_30_negative, columns=['Word', 'Frequency'])
else:
    df_top_negative = pd.DataFrame(columns=['Word','Frequency'])

if pos_id is not None:
    top_30_positive = top_words_for(pos_id)
    df_top_positive = pd.DataFrame(top_30_positive, columns=['Word', 'Frequency'])
else:
    df_top_positive = pd.DataFrame(columns=['Word','Frequency'])

if neu_id is not None:
    top_30_neutral = top_words_for(neu_id)
    df_top_neutral = pd.DataFrame(top_30_neutral, columns=['Word', 'Frequency'])
else:
    df_top_neutral = pd.DataFrame(columns=['Word','Frequency'])

plt.figure(figsize=(12,6))
sns.barplot(x="Word", y="Frequency", data=df_top_negative, palette="viridis")
plt.xticks(rotation='vertical')
plt.title("Top 30 Negative Words in Corpus")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(21,6), sharey=True)
sns.barplot(x="Word", y="Frequency", data=df_top_negative, palette="viridis", ax=axes[0])
axes[0].set_title("Top 30 Negative Words in Corpus")
axes[0].tick_params(axis='x', rotation=90)

sns.barplot(x="Word", y="Frequency", data=df_top_positive, palette="magma", ax=axes[1])
axes[1].set_title("Top 30 Positive Words in Corpus")
axes[1].tick_params(axis='x', rotation=90)

sns.barplot(x="Word", y="Frequency", data=df_top_neutral, palette="crest", ax=axes[2])
axes[2].set_title("Top 30 Neutral Words in Corpus")
axes[2].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()

# ------------------- Remove classes with less than 2 samples -------------------
counts = df['Target'].value_counts()
valid_classes = counts[counts >= 2].index
df = df[df['Target'].isin(valid_classes)]

# ------------------- CountVectorizer baseline -------------------
cv = CountVectorizer()
x = cv.fit_transform(df['Transformed_Text'])
y = df['Target'].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2, stratify=y
)

performance_data = []

# GaussianNB (dense only, memory heavy → wrap in try/except)
try:
    gnb = GaussianNB()
    gnb.fit(x_train.toarray(), y_train)
    y_pred1 = gnb.predict(x_test.toarray())
    acc1 = accuracy_score(y_test, y_pred1)
    prec1 = precision_score(y_test, y_pred1, average='macro', zero_division=0)
    print("\nGaussianNB")
    print(f"Accuracy : {acc1}")
    print(f"Precision Score : {prec1}")
    print(f"Confusion Matrix :\n{confusion_matrix(y_test, y_pred1)}")
    performance_data.append(("GaussianNB", acc1, prec1))
except MemoryError:
    print("Skipping GaussianNB due to memory constraints.")

# MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_pred2 = mnb.predict(x_test)
acc2 = accuracy_score(y_test, y_pred2)
prec2 = precision_score(y_test, y_pred2, average='macro', zero_division=0)
print("\nMultinomialNB")
print(f"Accuracy : {acc2}")
print(f"Precision Score : {prec2}")
print(f"Confusion Matrix :\n{confusion_matrix(y_test, y_pred2)}")
performance_data.append(("MultinomialNB", acc2, prec2))

# BernoulliNB
bnb = BernoulliNB()
bnb.fit(x_train, y_train)
y_pred3 = bnb.predict(x_test)
acc3 = accuracy_score(y_test, y_pred3)
prec3 = precision_score(y_test, y_pred3, average='macro', zero_division=0)
print("\nBernoulliNB")
print(f"Accuracy : {acc3}")
print(f"Precision Score : {prec3}")
print(f"Confusion Matrix :\n{confusion_matrix(y_test, y_pred3)}")
performance_data.append(("BernoulliNB", acc3, prec3))

# ------------------- TF-IDF models -------------------
tfidf = TfidfVectorizer()
x_tfidf = tfidf.fit_transform(df['Transformed_Text'])
y = df['Target'].values

x_train, x_test, y_train, y_test = train_test_split(
    x_tfidf, y, test_size=0.2, random_state=2, stratify=y
)

models = {
    "MultinomialNB": MultinomialNB(),
    "BernoulliNB": BernoulliNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    
}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"\n{name}")
    print(f"Accuracy : {acc}")
    print(f"Precision : {prec}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    performance_data.append((name, acc, prec))

# ------------------- Stacking -------------------
stack = StackingClassifier(
    estimators=[('mnb', MultinomialNB()), ('lr', LogisticRegression(solver='liblinear'))],
    final_estimator=LogisticRegression()
)

stack.fit(x_train, y_train)
y_pred_stack = stack.predict(x_test)

acc_stack = accuracy_score(y_test, y_pred_stack)
prec_stack = precision_score(y_test, y_pred_stack, average='macro', zero_division=0)

print("\nStackingClassifier")
print(f"Accuracy : {acc_stack}")
print(f"Precision : {prec_stack}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_stack)}")

performance_data.append(('StackingClassifier', acc_stack, prec_stack))

# ------------------- Performance Comparison -------------------
performance_df = pd.DataFrame(performance_data, columns=['Algorithm', 'Accuracy', 'Precision'])
performance_data_melted = pd.melt(performance_df, id_vars='Algorithm', var_name='Metric', value_name='Score')

sns.catplot(
    x='Algorithm', y='Score', hue='Metric',
    data=performance_data_melted, kind='bar', height=6
)
plt.xticks(rotation=45, ha='right')
plt.ylim(0.5, 1.05)
plt.title("Model Performance Comparison - Accuracy v/s Precision")
plt.show()

# ------------------- SAVE ARTIFACTS -------------------
os.makedirs("models", exist_ok=True)
pickle.dump(tfidf, open("models/vectorizer.pkl", "wb"))
pickle.dump(stack, open("models/model.pkl", "wb"))

print("Model and vectorizer saved successfully to models/.")