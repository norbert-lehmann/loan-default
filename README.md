## Predictive Profit Pioneers 😉

1. Joanna Sujecka
2. Piotr Łaciński
3. Marcin Płatek
4. Piotr Mielczarek
5. Mariusz Lenkiewicz
6. Norbert Lehmann



### Analiza danych wejściowych
__ispec_accuracy__ będzie podobne, jeżeli dane będą podobne.

You can use the describe() function in pandas to get a summary of

* the central tendency,
* dispersion,
* shape of a dataset’s distribution.

If the statistics for these two sets of values are significantly different, 
it suggests that the sets of values are different, otherwise they are similar.

https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a

Mocne niezbilansowanie próby w regresji logistycznej odnosi się do sytuacji, 
gdy liczba obserwacji w jednej z klas (np. klasy pozytywnej) znacznie przewyższa liczbę obserwacji
 w drugiej klasie (np. klasy negatywnej). Jest to częsty problem w analizie danych, 
zwłaszcza w przypadkach, gdzie zdarzenia rzadko występują, takie jak choroby rzadkie, wypadki, czy inne zdarzenia nietypowe.

Mocne niezbilansowanie próby może wpływać na skuteczność modelu regresji logistycznej, 
ponieważ model ten jest wrażliwy na proporcje klas w danych treningowych. 
Oto kilka potencjalnych problemów związanych z mocnym niezbilansowaniem próby w regresji logistycznej:

* __Złamane prawdopodobieństwo klasyfikacji:__ Model może być skłonny przypisywać 
obserwacjom klasyfikację jako dominującej klasy, ignorując mniej liczne klasy. 
W skrajnych przypadkach, model może zawsze przewidywać klasę, która występuje częściej, 
co prowadzi do pozornie wysokiej skuteczności modelu, ale w rzeczywistości może być bezużyteczny w identyfikowaniu rzadkich zdarzeń.

* __Niskie czułość modelu:__ Czułość modelu (true positive rate) może być niska, 
co oznacza, że model może mieć trudności w identyfikowaniu rzeczywistych przypadków z klasy rzadziej występującej.

* __Przyczynianie się do błędów:__ Mocne niezbilansowanie próby może przyczynić się do błędów w modelu, 
takich jak nadmierne dopasowanie do dominującej klasy, a zaniedbywanie klasy rzadkiej.

Aby radzić sobie z mocnym niezbilansowaniem próby w regresji logistycznej, można podjąć kilka działań:

* __Resampling danych:__ Można dostosować proporcje klas poprzez oversampling 
(zwiększanie liczby próbek z klasy rzadkiej) lub undersampling (zmniejszanie liczby próbek z dominującej klasy).

* __Używanie wag klas:__ W regresji logistycznej można przypisać różne wagi klasom, 
aby skompensować niezbilansowanie.

* __Zastosowanie innych algorytmów:__ W niektórych przypadkach, inne modele, takie
 jak algorytmy ensemble lub algorytmy drzew decyzyjnych, mogą być mniej wrażliwe na niezbilansowanie próby.

Ocena modelu za pomocą odpowiednich metryk: Przy ocenie skuteczności modelu należy uwzględnić metryki takie jak precision, recall, F1-score, krzywa ROC itp., które są bardziej odpowiednie dla niezbilansowanych danych niż prosta dokładność.
Warto zaznaczyć, że wybór odpowiedniej strategii zależy od konkretnego przypadku i charakterystyki danych.
