import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('simulated_email_dataset.csv')

# Create categorical feature for email length
df["email_length_cat"] = pd.qcut(df['email_length'],q=4, labels=['short', 'medium', 'long', 'very_long'])
print(df.head())
# Calculate prior probabilities
counts = df['label'].value_counts()
total = len(df)

#Prior Probabilties 
p_spam = counts['spam'] / total
p_ham = counts['ham'] / total 



def email_classify(email_length_cat, contains_free, time_of_day):
    # Likelihoods
    df_spam = df[df['label'] == 'spam'] 
    p_length_given_spam = len(df_spam[df_spam['email_length_cat'] == email_length_cat]) / counts['spam']
    p_free_given_spam = len(df_spam[df_spam['contains_free'] == contains_free]) / counts['spam']
    p_tod_given_spam = len(df_spam[df_spam['time_of_day'] == time_of_day]) / counts['spam']

    df_ham = df[df['label'] == 'ham'] 
    p_length_given_ham = len(df_ham[df_ham['email_length_cat'] == email_length_cat]) / counts['ham']
    p_free_given_ham = len(df_ham[df_ham['contains_free'] == contains_free]) / counts['ham']
    p_tod_given_ham = len(df_ham[df_ham['time_of_day'] == time_of_day]) / counts['ham']
    
    # Using Bayes theorem
    p_spam_given_features = (p_length_given_spam * p_free_given_spam * p_tod_given_spam * p_spam)
    p_ham_given_features = (p_length_given_ham * p_free_given_ham * p_tod_given_ham * p_ham)

    # Normalization
    total_prob = p_spam_given_features + p_ham_given_features
    
    p_spam_final = p_spam_given_features / total_prob
    p_ham_final = p_ham_given_features / total_prob
    
    print(f"{p_spam_final:.4f}")
    print(f"{p_ham_final:.4f}")
   
 
    if p_spam_final > p_ham_final:
        return "spam"
    else:
        return "ham"
    


result = email_classify('short', 0, 'morning')

print()
print("The email is classified as :", result)
