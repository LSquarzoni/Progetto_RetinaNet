#RetinaNet

In questo file andrò velocemente ad elencare il contenuto delle cartelle presenti nella directory work/RetinaNet:

.../Aurora_Dataset/ : All'interno è presente il dataset Himax ottenuto dalla collega Aurora di Giampietro a partire dalla
videocamera del nano drone, quindi formato da una serie di immagini in bianco e nero contenenti lattine e bottiglie, 
tutte annotate e pronte per un loro utilizzo nella object detection con bounding boxes; il formato delle annotazioni, 
inizialmente in XML, è stato convertito in json, in modo da essere utilizzato con Detectron2.
Sono presenti due script, un primo per valutare i modelli addestrati con dataset Himax, ed un secondo per svolgere il
training stesso, come indicato dal nome. Infine le cartelle "output..." contengono i risultati dei diversi processi
di addestramento, differenziati il meglio possibile attraverso la nomenclatura; elemento più importante all'interno di 
queste cartelle è il file "model_final.pth", all'interno del quale sono registrati i valori dei parametri del modello
dopo il training, utili quindi per utilizzare il modello dopo il suo addestramento, che sia per inferenza o per 
valutazione o anche per un ulteriore training.

.../COCOdataset2014/ : Contiene solo le tante cartelle di output ottenute con il pre-train del modello.

.../detectron2/ : Cartella di installazione della libreria di Detectron2, dove trovare il codice di ogni funzione e 
risorsa utilizzata nei diversi processi del progetto.

.../detectron2_backbone/ : Cartella scaricata a partire dal GitHub https://github.com/sxhxliang/detectron2_backbone,
grazie alla quale abbiamo sostituito la backbone iniziale di RetinaNet, ResNet50, con quella più leggera, 
MobileNetV2, mediante la scrittura di alcune stringhe di codice nel file YAML presente nella cartella Script_Python.

.../Im_samples/ : Contiene le immagini utilizzate per fare alcuni test di inferenza per i modelli addestrati, alcune
prese da Google e altre dai dataset di test; la cartella IM_TestFinale contiene il gruppo di immagini utilizzato per 
testare le potenzialità sia con dataset OpenImages che Himax. Infine, le cartelle di OUTPUT contengono gli esiti 
dell'inferenza svolta con tutti i modelli rilevanti durante lo sviluppo.

.../OpenImagesV6/ : In questa cartella ci sono le immagini di sole lattine e bottiglie prese dal dataset OpenImagesV6,
sia quelle di train che di valutazione che di test. Come di consueto sono presenti le cartelle di output di training.
I file .csv contengono i dati di annotazione di ogni immagine, servono nel momento di registrazione del dataset
prima di ogni training con questo dataset, poichè non è in formato coco.

.../OpenImagesV6_TinCan/ : Analoga alla cartella precedente, contiene però il dataset nel quale abbiamo aumentato
il numero di immagini di lattine mediante Data Image Augmentation, in modo da bilanciare il dataset stesso.

.../Script_Python/ : Cartella più importante, contiene quasi tutti gli script utilizzati per fare training, valutazione,
inferenza, dataloading e tentativi nella costruzione ddel modello; inoltre c'è il file YAML con cui costruire 
RetinaNet con MobileNet V2. Alcuni file si somigliano nel contenuto, dato che sono stati creati nel tempo, durante il 
proseguo del progetto. Il file più completo di inferenza è quello INFERENCE.py, il quale contiene anche del codice
per fare inferenza delle immagini di un'intera cartella e salvarle con nomi differenziati o fare la selezione dell'
immagine direttamente da terminale con gli adeguati parametri.

Per svolgere un training è necessario prima entrare nella cartella del dataset che si vuole utilizzare, sia per 
ordine che per effettiva necessità, visto che, ad esempio con gli script di training con OpenImages, gli indirizzi 
utlizzati nel dataloading sono relativi, non assoluti.

Il processo che ho seguito per creare ad addestrare il modello finale è bene o male spiegato nella tesi, però procedo in 
una sua veloce spiegazione: per prima cosa, a partire dal model zoo di detectron2, ho estratto il modello di RetinaNet
ResNet50 senza però caricare i suoi parametri pre-trainati, questo perchè, avendo modificato abbastanza la sua struttuta
non ci sarebbe più stato riscontro tra le architetture. Abbiamo quindi sostituito la backbone con una MobileNetV2 grazie
alla repository sopra indicata, ed infine ridotto la dimensione della head lasciando un solo layer convoluzionale nelle
due sottoreti. Ovviamente ognuno di questi passaggi intermedi ha richiesto tantissime prove di training per capire 
come funzionasse detectron2 e per trovare i parametri migliori di training. Una volta ottenuto il modello finale ho 
proceduto con il suo training completo, composto da pre-train con COCO2014 intero (80 classi), train con OpenImages V6 
con solo le 2 classi utili ed infine fine-tuning sequenziale con dataset Himax. Per risolvere alcuni problemi di catastrophic
forgetting ho eseguito parallelamente anche un fine-tuning combinato, utilizzando gli ultimi due dataset combinati.

Le pagine di codice sono sempre un po' commentate, non dovrebbe essere troppo difficile ambientarsi e capire.
Il codice è stato composto a partire dalle pagine esemplificative e di spiegazione fornite da detectron2 sul loro github, 
insieme ad altre parti estratte da altre repository, che mi sono servite nell'utilizzo ed implementazione del dataset
OpenImages.

Spero di essere stato il più esaustivo possibile, buon lavoro!