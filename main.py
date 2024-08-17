import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

""" Задача 1 """
df = pd.read_csv('IMDB_Dataset.csv')

print(df.head())

df_subset = df.sample(10)

df_subset.to_csv('manual_labeled_subset.csv', index=False)

""" Задача 2 """
positive_words = ['excellent', 'great', 'awesome', 'amazing', 'good']
negative_words = ['terrible', 'bad', 'awful', 'worst', 'poor']


def rule_based_labeling(review):
    review = review.lower()
    if any(word in review for word in positive_words):
        return 'positive'
    elif any(word in review for word in negative_words):
        return 'negative'
    else:
        return 'neutral'


df['rule_based_label'] = df['review'].apply(rule_based_labeling)

df_rule_based = df[['review', 'rule_based_label']]
df_rule_based.to_csv('rule_based_labeled.csv', index=False)

""" Задача 3 """
df_manual = pd.read_csv('manual_labeled_subset.csv')


def manual_labeling(review):
    print(f"Review: {review}")
    label = input("Enter label (positive/negative/neutral): ")
    return label


df_manual['manual_label'] = df_manual['review'].apply(manual_labeling)

df_manual.to_csv('manual_labeled.csv', index=False)

""" Задача 4 """
df_manual = pd.read_csv('manual_labeled.csv')
df_rule_based = pd.read_csv('rule_based_labeled.csv')

df_manual = df_manual.rename(columns={'manual_label': 'label'})
df_rule_based = df_rule_based.rename(columns={'rule_based_label': 'label'})

df_combined = pd.concat([df_manual[['review', 'label']],
                         df_rule_based[['review', 'label']]], ignore_index=True)

df_combined.to_csv('combined_labeled.csv', index=False)

""" Задача 5 """
df_combined = pd.read_csv('combined_labeled.csv')

X = df_combined['review']
y = df_combined['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=500, solver='liblinear')
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)

""" Задача 6 """
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность: {accuracy}')

print(classification_report(y_test, y_pred))
