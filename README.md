# TRAINING:
Di seguito la descrizione su come svolgere un generico training ed i dettagli riguardanti i parametri
utilizzati negli step di addestramento svolti durante lo svolgimento del progetto.

Per svolgere un training bisogna innanzitutto spostarsi nella cartella contenente il dataset desiderato, in modo
che il dataloader riesca a trovare i file di annotazione e le immagini costituenti il set di dati; così facendo inoltre,
i file di output del training verranno salvati nella cartella inerente.
Nelle cartelle Script_Python e Aurora_Dataset sono presenti i file necessari a svolgere un processo di training utilizzando
la libreria Detectron2; sono presenti alcuni file che presentano delle ripetizioni, questo perché durante lo sviluppo
ho svolto diversi test e cambiato alcuni elementi, capendo nel tempo come lavorare.

Esempio di codice a terminale: 
```
cd /home/lsquarzoni/work/RetinaNet/Aurora_Dataset
python /home/lsquarzoni/work/RetinaNet/Aurora_Dataset/Training_Himax.py 
```

Andiamo ad osservare, come esempio, il contenuto del file .../Aurora_Dataset/Training_Himax.py:

Come prima cosa vengono importate le risorse necessarie, per poi andare ad effettuare il DATALOADING del dataset
### Dataset COCO e Himax:
i dataset COCO e Himax sono in formato COCO, per questo motivo basta applicare la funzione register_coco_instances() per 
svolgere il dataloading. 
### Dataset OpenImagesV6:
il dataset OpenImagesV6 invece ha un formato differente, per questo motivo negli script 
di training con set OpenImages viene utilizzato un metodo di dataloading più lungo, che ho copiato da una utilissima
repository github: https://github.com/chriskhanhtran/object-detection-detectron2.

Successivamente bisogna costruire il modello:
per prima cosa si richiama il nodo di configurazioni attraverso cfg = get_cfg(),
poi si crea il modello a partire dal file YAML
### Modello leggero con backbone MobileNetV2:
per utilizzare il modello leggero con MobileNetV2 utilizzare 
``` 
cfg.merge_from_file("/home/lsquarzoni/work/RetinaNet/Script_Python/retinanet_mnv2.yaml") 
```
--> in questo caso bisogna anche settare cfg.MODEL.RETINANET.NUM_CONVS = 1 per rimuovere 3 dei layer della head.
### Modello con backbone ResNet50:
per utiliizare la backbone ResNet50
``` 
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml")) 
```
--> in questo caso settare cfg.MODEL.RESNETS.NORM = "BN" per evitare che la backbone sia freezata.

per caricare i weight ottenuti dopo un determinato training si utilizza cfg.MODEL.WEIGHTS = ".../model_final.pth".
Bisogna settare i dataset di train e di valutazione mediante cfg.DATASETS.TRAIN e cfg.DATASETS.TEST,
e anche tutti i possibili valori di training che vogliamo utilizzare:
io ho modificato solo alcuni dei parametri di default:
cfg.SOLVER.BASE_LR = 0.00005 è la base della learning rate che ho utilizzato per tutti i training (OpenImages e Himax)
cfg.SOLVER.MAX_ITER indica il numero di iterazioni che devono essere svolte

L'optimizer utilizzato in tutti i training è Adam, il quale è stato definito alla stringa 125 del file
.../detectron2/detectron2/solver/build.py

Il training vero e proprio si svolge facendo:
``` 
trainer = DefaultTrainer(cfg)         #Si costruisce il Trainer sulla base delle configurazioni fornite 
trainer.resume_or_load(resume=False)  #Se un training è stato interrotto, grazie a resume=True si riprende da quel punto 
trainer.train()                       #Inizio training 
```

Solitamente tutti i file di training contengono alcune stringhe di codice che, al termine del training stesso,
svolgeranno una valutazione utilizzando il dataset di valutazione fornito, che solitamente era quello del dataset con cui 
è stato svolto l'addestramento; esistono infatti diversi modi per svolgere una valutazione su Detectron2, per questo
motivo in alcune circostanze ne ho utilizzati di differenti.

Di seguito scrivo le diverse specifiche di training utilizzate nei diversi training svolti:
## TRAINING OPENIMAGESV6 (sia non bilanciato che bilanciato):
file .../Script_Python/TrainingOpenImages.py                    #non bilanciato
file .../Script_Python/TrainingOpenImages_Tincan_Augment.py     #bilanciato
``` 
cfg.SOLVER.BASE_LR = 0.00005  #5e-5 
cfg.SOLVER.MAX_ITER = 100000  #centomila --> unico training corposo 
```
## FINE TUNING SEQUENZIALE HIMAX:
file .../Aurora_Dataset/Training_Himax.py
``` 
cfg.SOLVER.BASE_LR = 0.00005  #5e-5 
cfg.SOLVER.MAX_ITER = 2500 --> training molto breve date le poche immagini 
```
in questo caso ho freezato tutta la backbone, sia nel caso ResNet che MobileNet, per farlo:
### Freezing MobileNetV2:
andare nel file .../detectron2_backbone/detectron2_backbone/backbone/mobilenet.py
stringa 8: decommentare (ricordarsi di ricommentare finito il training).
### Freezing ResNet50:
settare cfg.MODEL.BACKBONE.FREEZE_AT = 5 per freezare tutti e 5 gli stage della backbone.
## FINE TUNING COMBINATO OPENIMAGESV6+HIMAX:
file .../Script_Python/Training_OpenImages+DatasetDrone.py
``` 
cfg.DATASETS.TRAIN = ("AuroraDataset_train", "Bottle_TinCan_train",)  #2 dataset di train
cfg.SOLVER.BASE_LR = 0.00005  #5e-5 
cfg.SOLVER.MAX_ITER = 100000  #centomila --> unico training corposo 
```

## Modelli utili presenti nelle directory:
Nelle cartelle dei dataset sono presenti moltissimi checkpoints dei modelli addestrati, solo pochi però
sono stati usati in fase finale, di seguito elencherò quali siano quindi quelli di maggior rilevanza:
-RetinaNet-MobileNetV2 finale dataset Himax: .../Aurora_Dataset/outputFineTuning1_2.5k_LR5e-5_MNV2finale/model_final.pth
-RetinaNet-MobileNetV2 finale fine-tuning combinato: .../OpenImagesV6/outputTraining1_OpIm+Drone_100k_LR5e-5_MNV2finale/model_final.pth
-RetinaNet-MobileNetV2 finale dataset OpenImages: .../OpenImagesV6_TinCan/outputTraining1_100k_LR5e-5_MNV2finale/model_final.pth

# DATA AUGMENTATION:
la augmentation di immagini è sempre stata svolta nella fase di creazione dei dataset, mai applicata
ad inizio training; ad inizio training, di default, vangono applicate due funzioni di pre-processing:
ResizeShortestEdge e RandomFlip. 
Quando abbiamo provato ad usarne altre come RandomBrightness o RandomContrast i risultati sono peggiorati.
In ogni caso, per modificarle bisogna andare nel file .../detectron2/detectron2/engine/defaults.py e
alla stringa 548 inserire la volute funzioni (come esemplificato nel commento) da dare al DatasetMapper.

# EVALUATION:
Di seguito la descrizione su come ottenere una valutazione delle performance dei modelli ottenuti, 
utilizzando tutti i dataset utlizzati durante il progetto.

Come accennato nel file di training, solitamente al termine di un addestramento viene svolta una valutazione del modello
appena addestrato utilizzando il validation set dello stesso dataset; nel corso del progetto però è stato più
volte necessario svolgere valutazioni separate e con dataset differenti.
Per svolgere una valutazione spostarsi prima nella cartella del dataset di interesse.
Esempio di codice a terminale: 
``` 
cd /home/lsquarzoni/work/RetinaNet/OpenImagesV6_TinCan 
python /home/lsquarzoni/work/RetinaNet/Script_Python/Model_Evaluating_OpenImages.py 
```

## EVALUATION con dataset OPENIMAGESV6:
file .../Script_Python/Model_Evaluating_OpenImages.py
Come si vede nel file, anche per la valutazione con questo dataset è necessario innanzitutto fare dataloading
registrando i set bottle_tin_can_train e bottle_tin_can_val.
Anche in questo caso sarà necessario creare il modello, nello stesso modo visto nel file di training, assicurandosi 
di caricare i corretti weights, inerenti al modello di nostro interesse.
Il valore cfg.MODEL.RETINANET.SCORE_THRESH_TEST setta la soglia di confidenza sotto la quale le predizioni 
vengono scartate, di solito è stata tenuta a 0.5 in fase di valutazione.
A seconda che si voglia utilizzare il val set o il train set per effettuare la valutazione, bisogna specificare il
dataset nelle stringhe:
``` 
evaluator = COCOEvaluator("DATASET", cfg, False, output_dir=cfg.OUTPUT_DIR) 
val_loader = build_detection_test_loader(cfg, "DATASET") 
```
--> a termine del processo verranno stampati a terminale i risultati, distribuiti all'interno di una tabella, in formato
COCO metrics.

## EVALUATION con dataset HIMAX:
file .../Aurora_Dataset/Model_Evaluating_Himax.py
In questo caso il procedimento è analogo a quello con il primo dataset:
la scelta sarà tra AuroraDataset_train o AuroraDataset_val

# INFERENZA:
Di seguito la descrizione di come svolgere un generico processo di predizione utilizzando gli script
forniti.

Per svolgere una predizione è innanzitutto necessario avere alcune immagini in locale su cui la si voglia effettuare.
Nonostante i vari file presenti, è sufficiente utilizzare il seguente, .../Script_Python/INFERENCE.py, qualunque sia l'immagine,
collocandosi sempre nella cartella del dataset OpenImagesV6.
All'interno di questo infatti si vanno ad estrarre i metadati necessari per classificare le classi Bottle e Tin can.
Il file permette più di una modalità per svolgere inferenza:

## Fornendo l'indirizzo dell'immagine da linea di comando:
decommentare stringhe 121 e 122
commentare stringa 125
--> inserire il path assoluto del file come primo argomento:
Esemprio di codice da terminale:
``` 
cd /home/lsquarzoni/work/RetinaNet/OpenImagesV6
python /home/lsquarzoni/work/RetinaNet/Script_Python/INFERENCE.py work/RetinaNet/Im_samples/Im_fromGoogle/3bottiglie.jpg 
```

## Fornendo l'indirizzo dell'immagine direttamente nello script:
decommentare stringa 125
commentare stringhe 121 e 122
--> inserire il path assoluto del file all'interno di 
``` 
im = cv2.imread("PATH") 
```

A prescindere dalla modalità, successivamente sarà necessario costruire il modello, così come nel caso di evaluation e training.

Le seguenti stringhe andranno a svolgere l'inferenza:
``` 
outputs = predictor(im) 
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes) 

v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.5)
out = v.draw_instance_predictions(outputs["instances"].to("cpu")) 
```

Per salvare l'immagine di output bisogna inserire il path assoluto, con tanto di nome che si vuole dare al file, all'interno di:
``` 
cv2.imwrite('PATH ASSOLUTO OUTPUT', out.get_image()[:, :, ::-1]) 
```

Esiste una terza modalità per indicare le immagini su cui svolgere predizione:
## Fornendo un'intera cartella di immagini:
Un secondo file: .../Script_Python/Inference_OpenImages.py permette di svolgere inferenza su un'intera cartella di immagini.
Specificando il percorso assoluto della cartella alla stringa 154 infatti, un piccolo ciclo for andrà ad iterare lo svolgimento
delle predizioni, cambiando anche l'identificativo dell'immagine di output; importante prima di fare partire lo script,
sostituire il nome dei futuri file nella funzione nella stringa 163, sempre come percorso assoluto.
Anche in questo caso i passaggi imprescindibili sono la collocazione nella directory di OpenImagesV6 e la creazione del modello.