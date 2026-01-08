Ten plik nie jest używany w wersji pracy.

Został zastąpiony przez listingi kodów źródłowych:
- chapters/appendix_source_qh_policy_evaluation_pl.py
- chapters/appendix_source_qh_qlearning_pl.py

## Motywacja
- Standardowe RL zakłada dyskontowanie wykładnicze (preferencje spójne w czasie). W danych empirycznych często obserwuje się uprzedzenie teraźniejszości: natychmiastowe nagrody są nadmiernie ważone względem nagród odroczonych.
- Model quasi-hiperboliczny (\(\beta, \delta\)) jest prostym sposobem modelowania uprzedzenia teraźniejszości przy zachowaniu struktury, która jest wygodna obliczeniowo.

## Model preferencji i definicje wartości
- W chwili decyzyjnej \(t\) strumień nagród \((r_t, r_{t+1}, \dots)\) jest oceniany jako:
  \[
  r_t + \beta \sum_{k\ge 1} \delta^k \, r_{t+k}, \qquad \delta \in (0,1),\; \beta \in (0,1].
  \]
  Parametr \(\beta < 1\) opisuje uprzedzenie teraźniejszości; \(\beta=1\) redukuje model do standardowego dyskontowania wykładniczego.

Dla ustalonej polityki \(\pi\) w środowisku Markowskim definiuje się dwa obiekty:
- Wartość kontynuacyjna (wykładnicza) \(W^\pi(s)\):
  \[
  W^\pi(s)=\mathbb{E}_\pi\big[r(s,a)+\delta\, W^\pi(s')\mid s\big].
  \]
  Jest to standardowa funkcja wartości dla dyskontowania \(\delta\).
- Wartość quasi-hiperboliczna \(V^\pi(s)\):
  \[
  V^\pi(s)=\mathbb{E}_\pi\big[r(s,a)+\beta\delta\, W^\pi(s')\mid s\big].
  \]
  Interpretacja: nagroda „teraz” ma wagę 1, a suma wszystkich przyszłych nagród jest dodatkowo przeskalowana przez \(\beta\) i dalej dyskontowana wykładniczo przez \(\delta\).

Odpowiedniki akcyjno-stanowe:
- \(Q^C(s,a)\) — wartość sterowania kontynuacyjnego (wykładniczego):
  \[
  Q^C(s,a)=\mathbb{E}\big[r(s,a)+\delta \max_{a'} Q^C(s',a')\big].
  \]
- \(Q^I(s,a)\) — „cel natychmiastowego ja” (quasi-hiperboliczny):
  \[
  Q^I(s,a)= r(s,a)+\beta\delta\, \mathbb{E}\big[\max_{a'} Q^C(s',a')\big].
  \]

## Typy agentów i sterowanie
- Agent naiwny: planuje tak, jakby przyszłe „ja” realizowało bieżący plan (ignoruje przyszłą reoptymalizację). W praktyce często wybiera akcje na podstawie kryteriów wykładniczych, ale ocena zachowania ujawnia uprzedzenie teraźniejszości.
- Agent wyrafinowany (sophisticated): przewiduje przyszłą reoptymalizację i wybiera strategię stanowiącą równowagę (subgame-perfect equilibrium) pomiędzy kolejnymi „ja”. W ujęciu tablicowym odpowiada to wyborowi akcji przez maksymalizację \(Q^I\) przy jednoczesnym uczeniu \(Q^C\) jako komponentu bootstrapującego.

## Algorytmy

### 1) Ocena polityki (QH-TD(0))
- Uczymy \(W^\pi\) standardowym TD(0):
  \[
  W(s_t) \leftarrow W(s_t) + \alpha_{\text{lr}} \big[r_t + \delta W(s_{t+1}) - W(s_t)\big].
  \]
- Wartość quasi-hiperboliczną można raportować „w locie”:
  \[
  V_t(s_t) = r_t + \beta\delta\, W(s_{t+1}).
  \]
- Wersje z śladami (TD(\(\lambda\))) uzyskuje się analogicznie, używając \(\delta\) w zaniku śladu.

### 2) Sterowanie off-policy (QH-Q-learning, wersja wyrafinowana)
- Aktualizacja „głowy kontynuacyjnej” jak w standardowym Q-learningu:
  \[
  Q^C(s_t,a_t) \leftarrow Q^C(s_t,a_t) + \alpha_{\text{lr}} \big[r_t + \delta \max_{a'} Q^C(s_{t+1},a') - Q^C(s_t,a_t)\big].
  \]
- Cel natychmiastowego „ja” do selekcji akcji:
  \[
  Q^I(s_t,a) = r_t + \beta\delta\, \max_{a'} Q^C(s_{t+1},a').
  \]
- Działanie: zachłannie lub \(\varepsilon\)-zachłannie względem \(Q^I\); uczenie dotyczy tylko \(Q^C\), co sprzyja stabilności.

### 3) Sterowanie on-policy (QH-SARSA)
- Zastępujemy operator \(\max\) faktycznie wykonaną akcją \(a_{t+1}\) zarówno w aktualizacji \(Q^C\), jak i w definicji celu \(Q^I\).

### 4) Actor--critic (QH-AC)
- Krytyk uczy \(W^\pi\) standardowym TD z dyskontem \(\delta\).
- Aktor optymalizuje cel oparty o przewagę zbudowaną na:
  \[
  r_t + \beta\delta\, W(s_{t+1}).
  \]
  W policy gradient pojawiają się zwroty:
  \[
  G_t^{\beta,\delta} = r_t + \beta \sum_{k\ge 1} \delta^k r_{t+k}.
  \]

## Własności teoretyczne (poziom ogólny, przypadek tablicowy)
- Operator kontynuacyjny \(T_\delta\) jest kontrakcją o współczynniku \(\delta\). W konsekwencji tablicowe TD i Q-learning dla \(W\) lub \(Q^C\) dziedziczą standardowe gwarancje zbieżności (przy typowych założeniach: dostateczna eksploracja, kroki Robbins--Monro, ograniczone nagrody).
- Wartości natychmiastowe (\(V\), \(Q^I\)) są funkcjami algebraicznymi rozwiązania kontynuacyjnego; „uczalność” sprowadza się do problemu \(\delta\)-dyskontowanego.
- Dla agentów wyrafinowanych w stacjonarnych środowiskach istnieją stacjonarne równowagi Markowskie; działanie względem \(Q^I\) przy uczeniu \(Q^C\) można interpretować jako obliczanie takiej równowagi w skończonych MDP.

## Wskazówki praktyczne
- \(\beta\) kontroluje siłę uprzedzenia teraźniejszości; \(\beta=1\) daje standardowe RL.
- Architektury z wspólnym „trzonem” i dwiema „głowami” (kontynuacyjną i natychmiastową) są wygodne w aproksymacji funkcji; bootstrapping TD jest potrzebny tylko dla komponentu kontynuacyjnego.
- Dla \(Q^C\) warto stosować sieci docelowe (target networks) i Double Q-learning w celu ograniczenia przeszacowania; parametr \(\beta\) jedynie przeskalowuje bootstrap w \(Q^I\).
- W raportowaniu warto podawać zarówno zwroty wykładnicze (\(W\)), jak i quasi-hiperboliczne (\(V\)), aby ujawniać odwrócenia preferencji i odkładanie działań.

## Zjawiska empiryczne
- Odwrócenia preferencji i prokrastynacja: agent planuje działanie „później”, ale w momencie działania zmienia decyzję.
- Potrzeba prekomitmentu: agent wyrafinowany może poświęcić bieżącą korzyść, aby ograniczyć przyszłe wybory.
- Różnice jakości: agenci naiwni i wyrafinowani mogą osiągać różne wyniki długoterminowe przy tych samych \((\beta,\delta)\).

## Ograniczenia i rozszerzenia
- Bogatsze dyskontowanie (w pełni hiperboliczne) może lepiej opisywać zachowania, ale jest trudniejsze do stabilnego uczenia.
- Częściowa wyrafinowaność (tylko częściowe przewidywanie przyszłej reoptymalizacji) prowadzi do polityk pośrednich.
- W przestrzeniach ciągłych konieczna jest aproksymacja funkcji; stabilność podlega typowym ograniczeniom deep RL.

## Związek z implementacją w repozytorium
- Rozkład na komponent kontynuacyjny i natychmiastowy jest zgodny z implementacjami: uczenie \(W\) (ocena polityki) oraz uczenie \(Q^C\) i działanie przez \(Q^I\) (sterowanie).
- W testach warto rozróżniać przypadek \(\beta=1\) (redukcja do standardowego RL) i \(\beta<1\), a także weryfikować spójność oceny/zbieżności.

## Najważniejsze wnioski
- Uprzedzenie teraźniejszości w RL można implementować przez rozdzielenie uczenia komponentu kontynuacyjnego (\(\delta\)-dyskontowanego) od celu natychmiastowego, w którym przyszłość jest dodatkowo ważona przez \(\beta\).
- W ustawieniach tablicowych można wykorzystywać standardowe TD/Q-learning dla komponentu kontynuacyjnego, a quasi-hiperboliczny cel wyprowadzać algebraicznie.
- Ramy quasi-hiperboliczne łączą modele preferencji czasowych z praktycznymi algorytmami RL w sposób stosunkowo prosty obliczeniowo.
