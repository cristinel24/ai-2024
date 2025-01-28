# Catology - Sistem de clasificare a pisicilor după rasă

Acest proiect își propune să implementeze un sistem AI care să identifice rasa unei pisici pe baza unor atribute oferite de utilizator, inclusiv descrieri în limbaj natural. Ca funcționalități principale, sistemul:

1. **Traduce și adaugă instanțe noi** în setul de date disponibil [aici](https://data.mendeley.com/datasets/ht5p5pg7b7/1).  
2. **Citește** o descriere în limbaj natural a unei pisici, **extrage** atributele relevante și **identifică** rasa.  
3. **Generează** o descriere în limbaj natural pentru o rasă de pisici folosind clasificatorii antrenați.
---

## Echipa
- `Andrei Cristian-George`
- `Balan Călin`
- `Rotariu George-Flavian`

---

## Structura Proiectului

Proiectul este organizat în mai multe module Python, fiecare având responsabilități clare:

- **`main.py`** – Aici se încarcă datele, se pregătește modelul și se rulează pașii de inferență: traducerea textului, extragerea atributelor din descriere, prezicerea rasei.
- **`mlp.model.py`** – Clasa `MLPModel` care implementează rețeaua neuronală multi-layer perceptron. Include metode pentru antrenare, salvare/încărcare a modelului, precum și logica de forward/backward propagation.
- **`mlp.base.py`** – Clasa de bază `BaseModel`, care conține funcțiile utile de _softmax_, _relu_ și calculul pentru loss-ul de tip _cross-entropy_.
- **`engine.utils.py`** – Funcționalități de transformare a atributelor non-numerice în numerice (folosind `LabelEncoder`) și alte utilitare.
- **`engine.text_processing.py`** – Conține logica de citire a textului, detectarea limbii, traducerea în engleză, extragerea atributelor stilometrice (count cuvinte/ caractere), înlocuirea cu sinonime/hiperonime/antonime, extragerea cuvintelor cheie, generarea de descrieri cu GPT și funcția principală de parsare a textului în atribute tipice pisicilor.
- **`engine.statistics.py`** – Funcții pentru analiza setului de date (ex: distribuția instanțelor pe rase, statistici comportamentale, plot-uri).
- **`engine.processing.py`** – Etape de pre-procesare (ex: citirea dataset-ului, curățarea datelor lipsă, SMOTE pentru reechilibrarea claselor, adăugarea atributelor „Color” și „Pattern”).
- **`engine.plots.py`** – Funcții pentru generarea și afișarea matricilor de corelare, histograme etc.
- **`engine.constants.py`** – Conține path-urile și constantele de bază folosite în proiect.

---

# Funcționalități

- **Procesarea setului de date**  
   - Încărcarea CSV-ului, curățarea valorilor lipsă și eliminarea duplicatelor.  
   - Extinderea datelor cu noi atribute (`Color` și `Pattern`) pentru a îmbunătăți acuratețea.  

- **Echilibrarea claselor**  
   - Folosirea `SMOTE` pentru a crește numărul de instanțe din clasele minoritare și a evita overfitting-ul.  

- **Implementare MLP și Random Initialization**  
   - Crearea unui `Multi-Layer Perceptron` (MLP) cu un layer ascuns configurabil (implicit 100 de neuroni).  
   - Rețeaua folosește ReLU în stratul ascuns, softmax în stratul de ieșire și backpropagation cu _cross-entropy_. De asemenea, afișăm un grafic care ilustrează descreșterea funcției de cross entropie.  

- **Antrenarea inițială**  
   - Datele sunt împărțite în `train_data` (80%) și `test_data` (20%).  
   - S-a rulat un prim antrenament cu ~50 de epoci, pentru un baseline al acurateței.  

- **Optimizarea rețelei MLP**  
   - Implementare **early stopping** bazat pe reducerea dinamică a ratei de învățare.  
   - Testarea diferitilor hiperparametri (dimensiunea stratului ascuns, learning rate, batch size).  

- **Extragerea atributelor relevante din text**  
   - Implementearea unui modul **NLP** care:
     - Detectează limba (`langdetect`).
     - Traduce textul în engleză (`googletrans`).
     - Tokenizează și identifică sinonime/hiperonime/antonime (via `wordnet`).
     - Parsează textul pentru a găsi vârsta, sexul, culoarea, pattern-ul, trăsăturile comportamentale (ex: timid, agresiv, afectuos etc.).  

- **Prezicerea rasei pe baza atributelor**  
   - Formarea unui vector de intrare după parsare (ex: `[Sexe, Age, Nombre, Logement, Zone, Ext, Obs, ... Color, Pattern]`).  
   - Introducerea vectorului în MLP pentru a obține probabilitatea fiecărei rase. Rasa cu probabilitatea maximă este selectată.  

- **Evaluarea performanței**  
   - La finalul antrenamentului se calculează acuratețea pe setul de test. Modelul e salvat în `mlp_model.pkl`.  
   - Se pot atinge acurateți de `50-60%` (în funcție de date și finețea cu care se disting rasele).  

- **Generarea descrierilor de rasă**  
   - Folosirea unui model GPT (prin API) pentru a genera texte scurte, în română, despre fiecare rasă.  

- **Analiză stilometrică**  
   - Calcularea numărului de cuvinte, caractere, frecvența cuvintelor, etc.  

- **Înlocuirea cuvintelor**  
   - Înlocuirea a ~20% din cuvinte cu sinonime/hiperonime, testând robustețea la variații lingvistice.  

---

## Rezultate finale și concluzii

- **Acuratețe**  
   - Modelul MLP atinge ~60%+ acuratețe. Poate fi îmbunătățit cu mai multe date și atribute mai relevante.

- **Robustețe la text**  
   - Sistemul identifică atributele chiar și cu modificări lexicale moderate.
   - Implementarea unui sistem de intensificatori ai atributelor (ex: `foarte -> +3`, `deloc -> -2` etc.)

- **Descrieri generate**  
   - GPT generează text coerent în limba română cu detalii despre rase (ex. Bengal, Birman, etc.).  

- **Extensibilitate**  
   - Arhitectura permite adăugarea de noi atribute sau modele mai complexe.  

---

## Contribuții
De-a lungul acestui semestru, am lucrat împreună în cea mai mare parte a timpului pentru a ne atinge obiectivele, dar, dacă ar fi să menționăm contribuțiile speciale pe care fiecare dintre noi le-a avut la acest proiect, ar fi cam așa:

- **Rotariu George-Flavian**: Implementare MLP, logica de antrenare și salvare/încărcare a modelului.  
- **Balan Călin**: Pre-procesarea setului de date (SMOTE, atribute `Color`/`Pattern`), statistici, plot-uri.  
- **Andrei Cristian-George**: Parsare text, traducere, sinonime/hiperonime, intensificatori, integrarea temelor, integrare GPT pentru descrieri.  

---

## Referințe

1. [GitHub - Romanian-NLP-tools](https://github.com/Alegzandra/Romanian-NLP-tools)  
2. [GPT-Neo Romanian 780M](https://huggingface.co/dumitrescustefan/gpt-neo-romanian-780m)  
3. [Dataset pisici](https://data.mendeley.com/datasets/ht5p5pg7b7/1)  
4. [Googletrans Docs](https://py-googletrans.readthedocs.io/en/latest/)  
5. [NLTK](https://www.nltk.org/)  

---