import numpy as np
import pandas as pd
import os
import re

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# sklearn.neighbors import KNeighborsClassifier

#scieżka do projektu, bez uruchamia się w bieżącym katalogu
#os.chdir('/home/pawel/Dokumenty/Dydakt/Zima2023/Fintech_classification/classification_project')

# wczytaj dane
#parquest - columnar, compressed storage file format that is optimized for use with big data processing frameworks
df0 = pd.read_parquet('data/train_data_0.pq').astype('int32')
df1 = pd.read_parquet('data/train_data_1.pq').astype('int32')
df = pd.concat([df0, df1])

dtypes_dict = {'pclose_flag': 'int8', 'fclose_flag': 'int8',
               'is_zero_util': 'int8', 'is_zero_over2limit': 'int8', 'is_zero_maxover2limit': 'int8'}
df = df.astype(dtypes_dict)

##df.to_csv('output.csv', index=False)
##print("dane wejściowe zapisane")

target = pd.read_csv('data/target.csv').astype({'flag': 'int8'})


# JOIN LEFT - zmienne wejsciowe + zmienna wyjsciowa
df = df.merge(target, how='left', on='id')

# czynnosci techniczne: oczyszczenie pamieci, zmiana nazwy zmiennej wyjsciowej ('Y')
del(df0)
del(df1)
df = df.rename(columns={'flag': 'Y'})

#ustalenie podziału danych wejściowych na dane treningowe i testowe. 75% danych uzytych do trenowania.
#Losowanie danych na podstawie wartości random_state
df_train, df_test = train_test_split(df, train_size=0.75, random_state=15767)
# stratify? walidacja krzyżowa?

print('W zbiorze do treningu', len(df_train), "a testowym", len(df_test), "obs.")

# wczytaj z pliku specyfikacje (lista list; pomin linie bez zawartości)
with open('specs_logit.txt', 'r') as f:
    specifications = [i.split() for i in f.readlines() if re.findall(r'\w+', i)]

accuracy_list = []   # prostacki sposób przechowywania wyników

#ispec_accuracy będzie podobne, jeżeli dane będą podobne
#Check Descriptive Statistics: You can use the describe() function in pandas to get a summary of
#-the central tendency,
#-dispersion,
#-shape of a dataset’s distribution.
#If the statistics for these two sets of values are significantly different,
#it suggests that the sets of values are different, otherwise they are similar.
##print("Statystyczny opis danych")
##print(df[['rn','pre_loans_credit_limit']].describe())
##print(df[['is_zero_loans3060', 'is_zero_loans6090', 'is_zero_loans90']].describe())

print("Wizualizacja danych")
#Wizualizacja danych, żeby porównać dystrybucję
##plt.hist(df['rn'], alpha=0.4, label='rn')
##plt.hist(df['pre_loans_credit_limit'], alpha=0.5, label='pre_loans_credit_limit')
##plt.hist(df['is_zero_loans3060'], alpha=0.6, label='is_zero_loans3060')
##plt.hist(df['is_zero_loans6090'], alpha=0.7, label='is_zero_loans6090')
##plt.hist(df['is_zero_loans90'], alpha=0.8, label='is_zero_loans90')
#pre_loans530 pre_till_fclose
#plt.hist(df['is_zero_loans90'], alpha=0.8, label='is_zero_loans90')
##plt.legend(loc='upper right')
##plt.show()

# LogisticRegression - loop over specifications
print("looping over specifications")
for ispec in specifications:
    print(ispec)
    mymodel = LogisticRegression()
    X_train, X_test = df_train[ispec], df_test[ispec]
    mymodel.fit(X_train, df_train['Y'])
    y_predicted_test = mymodel.predict(X_test)
    # TODO: mocne niezbilansowanie próby (<5% obs. Y==1)
    # y_predicted_test = (mymodel.predict_proba(X_test) > df_train.Y.mean())
    ispec_accuracy = accuracy_score(df_test['Y'], y_predicted_test)
    print("Logit", ispec, "\nAccuracy", round(ispec_accuracy, 4))
    accuracy_list.append(ispec_accuracy)
    # TODO: dodaj inne miary jakości modelu - F1, balanced accuracy


# TODO: dodaj inne metody, np. GradientBoosting, XGBoost
# np. https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html


# RandomForest - przykład metody z hiperparametrem(ami)

# jeden wariant specyfikacji (przykład bez znaczenia)
ispec = ['rn', 'pre_since_opened', 'pre_loans_credit_limit', 'pre_loans_total_overdue',
         'is_zero_util', 'pre_loans530', 'pre_loans3060', 'pre_loans6090', 'pre_loans90',
         'pre_loans_max_overdue_sum', 'enc_loans_credit_status', 'pre_till_pclose']

# jakie głębokości lasu losowego rozważymy
tree_maxdepths_versions = [3, 4, 5, 6]

for i_maxdepth in tree_maxdepths_versions:
    # HIPERPRAMETRY
    # max depth - z pętli (warianty)
    # min_sample_leaf - zapobieganie przeuczeniu
    mymodel = RandomForestClassifier(max_depth=i_maxdepth, min_samples_leaf=20)
    X_train, X_test = df_train[ispec], df_test[ispec]
    mymodel.fit(X_train, df_train['Y'])
    y_predicted_test = mymodel.predict(X_test)
    # TODO: mocne niezbilansowanie próby (<5% obs. Y==1)
    # y_predicted_test = (mymodel.predict_proba(X_test) > df_train.Y.mean())
    ispec_accuracy = accuracy_score(df_test['Y'], y_predicted_test)
    print("Las losowy max depth", i_maxdepth, "\nAccuracy", round(ispec_accuracy, 4))
    # accuracy_list.append(ispec_accuracy)
    # TODO: dodaj inne miary jakości modelu - F1, balanced accuracy


# TODO: walidacja krzyżowa? dobór hiperparametrów

# TODO: feature importance?

# TODO: inne metody radzenia sobie z niezbilansowanym Y (upsampling, downsampling)

print("Koniec obliczeń")
