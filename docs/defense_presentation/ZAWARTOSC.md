# Zawartość prezentacji obronowej

Ten dokument opisuje szczegółową zawartość 10-minutowej prezentacji obronowej.

## Struktura slajdów

### 1. Strona tytułowa (1 slajd)
- Tytuł pracy: "Algorytmy oparte o wzmocnione nauczanie maszynowe w dyskontowanych modelach markowskich"
- Autor: Michał Wrona
- Uczelnia: Politechnika Wrocławska, Wydział Matematyki
- Promotor: prof. dr hab. inż. Anna Jaśkiewicz
- Data

---

## SEKCJA: WSTĘP (7 slajdów)

### 2. Plan prezentacji (1 slajd)
- Automatycznie generowany spis treści z czterech sekcji

### 3. Motywacja wyboru tematu (1 slajd)
**Zawartość:**
- Wprowadzenie do Reinforcement Learning
- Problem tradycyjnego wykładniczego dyskontowania
- Dyskontowanie quasi-hiperboliczne jako rozwiązanie
- Zastosowania praktyczne

### 4-5. Wprowadzenie oznaczeń matematycznych (2 slajdy)
**Slajd 4:**
- Definicja MDP: $M = (S, A, q, r, \alpha, \beta)$
- Objaśnienie wszystkich symboli
- Funkcja dyskontująca $d(t)$
- Skumulowana nagroda

**Slajd 5:**
- Typy decydentów (według parametru $\alpha$)
- Pojęcie polityki i funkcji wartości

### 6-9. Kluczowe wyniki z literatury (4 slajdy)
**Slajd 6: Twierdzenie o redukcji polityk**
- Twierdzenie 1 z Eshwar et al. (2025)
- Redukcja do pary $(\mu, \phi_s)$
- Odniesienie do dowodu (bez szczegółów)

**Slajd 7: Operator Bellmana**
- Lemat o kontrakcji
- Implikacje dla zbieżności
- Odniesienia: Puterman (2014), Kallenberg (2022)

**Slajd 8: Istnienie optymalnej polityki**
- Twierdzenie 2 z Eshwar et al. (2025)
- Deterministyczna optymalna para
- Znaczenie dla algorytmów

**Slajd 9: Algorytmy uczenia ze wzmocnieniem**
- QH Policy Evaluation
- QH Q-Learning
- Twierdzenie o zbieżności
- Odniesienia: Eshwar et al. (2025), Borkar (2008)

---

## SEKCJA: WYNIKI WŁASNE (6 slajdów)

### 10. Środowiska testowe (1 slajd)
- MDP z dwoma stanami
- Problem zarządzania zapasami
- Parametry eksperymentów ($\alpha = 0.5$, $\beta = 0.9$)

### 11-12. MDP z dwoma stanami (2 slajdy)
**Slajd 11: Model**
- Definicja stanów i akcji
- Funkcja nagrody
- Diagram przejść (grafika z pracy)
- Testowane polityki

**Slajd 12: Wyniki**
- Weryfikacja z obliczeniami analitycznymi
- Niespójność czasowa
- Warunki Robbinsa-Monro

### 13-14. Problem zarządzania zapasami (2 slajdy)
**Slajd 13: Model**
- Opis problemu biznesowego
- Parametry środowiska
- Cele eksperymentu

**Slajd 14: Wyniki**
- Różnice w strategiach (QH vs wykładniczy)
- Zbieżność algorytmu
- Porównanie decydentów

### 15. Implementacja i metodologia (1 slajd)
- Narzędzia (Python, NumPy, etc.)
- Struktura kodu
- Metody walidacji

---

## SEKCJA: PODSUMOWANIE (7 slajdów + slajd końcowy)

### 16. Wnioski (1 slajd)
**Osiągnięcia:**
- Teoretyczne (analiza polityk, dowody zbieżności)
- Praktyczne (implementacja algorytmów)
- Poznawcze (zrozumienie niespójności czasowej)

### 17. Niespójność czasowa (1 slajd)
- Kluczowe odkrycie dla $\alpha < 1$
- Implikacje praktyczne
- Przykład studenta uczącego się do egzaminu

### 18-19. Dalsze prace (2 slajdy)
**Slajd 18: Kierunki rozwoju**
- Rozszerzenie teorii (decydenci wyrafinowani, gry wieloagentowe)
- Zaawansowane algorytmy (Deep Q-Learning, Actor-Critic)
- Zastosowania praktyczne

**Slajd 19: Wyzwania badawcze**
- Estymacja parametrów z danych
- Adaptacja online
- POMDP z QH
- Możliwości interdyscyplinarne

### 20-21. Bibliografia (2 slajdy)
- Eshwar (2025, 2024)
- Jaśkiewicz & Nowak (2021)
- Borkar (2008)
- Puterman (2014)
- Kallenberg (2022)
- Sutton & Barto (2018)
- Laibson (1997)
- Bertsekas (2019)

### 22. Slajd końcowy (1 slajd)
- "Dziękuję za uwagę"
- "Pytania?"
- Dane kontaktowe

---

## Łączna liczba slajdów: ~22

## Oszacowany czas prezentacji

- Strona tytułowa: 0.5 min
- Wstęp: 3.5 min
- Wyniki własne: 4 min
- Podsumowanie: 2 min
- **RAZEM: ~10 minut**

## Kluczowe cechy prezentacji

1. **Zgodność z wymaganiami:** Wszystkie elementy z zadania są uwzględnione
2. **Struktura logiczna:** Od motywacji przez teorię do wyników i wniosków
3. **Odniesienia bibliograficzne:** Wszystkie twierdzenia mają cytowania bez prezentacji dowodów
4. **Wizualizacje:** Wykorzystanie diagramów z pracy dyplomowej
5. **Profesjonalny wygląd:** Temat Madrid, schemat kolorów beaver
6. **Język polski:** Cała prezentacja po polsku zgodnie z wymogami
7. **Format 16:9:** Nowoczesny format ekranu panoramicznego

## Uwagi dodatkowe

- Prezentacja nie zawiera dowodów matematycznych (tylko stwierdzenia twierdzeń)
- Wszystkie wzory matematyczne są czytelne i dobrze sformatowane
- Bibliografia zawiera pełne odniesienia do wykorzystanych prac
- Struktura pozwala na elastyczne dostosowanie czasu (można pominąć niektóre szczegóły)
- Prezentacja jest samodzielna i nie wymaga dodatkowych materiałów
