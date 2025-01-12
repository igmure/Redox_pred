### Redox_pred: Przewidywanie potencjałów redoks za pomocą uczenia maszynowego  

Potencjały redoks dla reakcji jednoelektronowych odgrywają kluczową rolę w wielu dziedzinach, takich jak kataliza, magazynowanie energii czy procesy biologiczne. Tradycyjne metody oparte na teorii funkcjonałów gęstości (DFT) pozwalają na dokładne przewidywanie potencjałów redoks, lecz są czasochłonne i kosztowne obliczeniowo, dlatego coraz częściej stosuje się metody uczenia maszynowego.  

W niniejszym projekcie zbadano wpływ różnych deskryptorów molekularnych na przewidywanie potencjału redoks z wykorzystaniem metod uczenia maszynowego. Analiza przeprowadzona na zbiorze **ROP313** wykazała, że dodanie informacji o **ładunku cząsteczki** oraz **liczbie wolnych elektronów** znacząco poprawia jakość predykcji.  

Modele oparte na **deskryptorach 3D** nie przyniosły poprawy wyników, co przypisano ograniczonej liczbie danych. Najlepsze rezultaty osiągnął model **Random Forest**, wskazując na potrzebę stosowania metod uwzględniających nieliniowość danych i możliwość odrzucania mniej istotnych cech.  

Wyniki podkreślają znaczenie wyboru deskryptorów w modelowaniu właściwości elektrochemicznych i sugerują kierunki dalszych badań, takie jak testowanie zaawansowanych modeli ML i innych zbiorów danych.  

---

### Pliki projektu  

- **Szczegółowy przebieg eksperymentów**: [model.ipynb](#)  
  Zawiera kompletny przebieg eksperymentów i proces trenowania modeli.  

- **Wizualizacja danych**: [VALUES_SMILES.ipynb](#)  
  Obejmuje wizualizacje zbioru danych oraz analizę deskryptorów.  
