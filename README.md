# Wykrywanie obszaru i punktów kluczowych kortu tenisowego


## Problem


## Założenia


## Ogólny schemat i opis procesu


## Szczegółowy opis algorytmu przeszukiwania kortu tenisowego i  znajdowania punktów kluczowych


## Korekty

Znalezione segmenty ograniczające obszar kortu i wyznaczające kluczowe jego punkty wymagają korekty z uwagi na szum uwzględniony w trakcie trawersowania [tutaj odnosnik] - ma to na celu zwiększenie jakości oraz poprawności wyników detekcji.

### Segmenty linii bocznych kortu

1. docinanie kolekcji segmentów wg punktu przecięcia z kolekcją przeciwległą

2. eliminacja segmentów odstających wg ich położenia względem osi OX

3. podmiana segmentów odstajacych względem atrybutów linii - tj. współczynnika kierunkowego (slope) i wyrazu wolnego (intercept)

4. uzupełnienie ewentualnych luk w kolekcji segmentów
    - procesowanie punktu `2` mogło skutkować potencjalnym wybrakowaniem
    - dlatego z uwagi na zachowanie ciągłości znalezione braki uzupełniane są wg punktów sąsiadujących ze sobą segmentów, pomiędzy którymi taka luka powstała

### Segmenty horyzontalne

### Dopasowanie końców segmentów do punktów przecięcia
Dotyczy zarówno segmentów horyzontalnych kortu jak i linii bocznych - chodzi o to, aby ograniczyć kolekcję segmentów dokładnie do punktu przecięcia. Można wyróżnić dwa typy sytuacji:

1. wydłużenie segmentu do punktu przecięcia - tam, gdzie obie kolekcje się wizualnie nie przecinają

2. skrócenie segmentu - gdy segment zaczyna się lub kończy poza punktem przecięcia


Należy jednak pamiętać, że kwestia ta ma przede wszystkim znaczenie wizualne. Funkcja `find_line_segments_intersection` wyznacza punkt przecięcia prostych odpowiadających segmentom, a nie tylko ich fizycznych odcinków. Jeżeli punkt przecięcia należy do obu segmentów, funkcja zwraca go bezpośrednio. W przeciwnym razie spośród wszystkich znalezionych przecięć wybiera to, dla którego odległość między środkami odpowiadających segmentów jest najmniejsza. Dzięki temu może znaleźć przecięcie również wtedy, gdy segmenty nie stykają się bezpośrednio na obrazie.


## Testy


## Ewaluacja


## Deployment

