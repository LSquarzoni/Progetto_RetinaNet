#RetinaNet

File e script necessari a lavorare su RetinaNet su libreria Detectron2.

Gli script presenti in questa cartella sono necessari a svolgere alcune azioni di base, come dataloading di dataset COCO2014 e OpenImagesV6 di sole bottiglie e lattine,
training, valutazione di modelli ed infine inferenza. Il file retinanet_mnv2.yaml serve a costruire l'architettura 
RetinaNet con backbone MobileNetV2, come mostrato nel codice.

Per quanto riguarda il codice per svolgere le stesse azioni con il dataset Himax delle immagini di drone, questo si trova
all'interno della cartella Aurora_Dataset.

Le cartelle detectron2 e detectron2_backbone sono state scaricate direttamente da github e presentano il codice 
della libreria utilizzata, all'interno del quale sono state applicate alcune piccole modifiche in fase progettuale, con 
lo scopo di esplorare il tool e comprenderlo. 

Infine, le cartelle COCO e OpenImages, contengono i rispettivi dataset, con immagini ed annotazioni, ed i risultati dei tanti processi di training svolti
nelle diverse configurazioni, tutti differenziati il meglio possibile attraverso i nomi.