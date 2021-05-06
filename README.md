# Parrot & toucan
NN for the detection task parrot and toucan.

For marking up images, I used [supervise.ly](https://supervise.ly/).

Ok, this two project.

## Project1_yolo5, [YOLO5](https://github.com/ultralytics/yolov5)

Task to detected toucan or parrot on image:

- Train data have image in forest parrot and toucan
- for train use train_test_split 
- 150 epoch
- yolo5m
  
Who is who:
This Parrot             |  This guy or not, is Toucan
:-------------------------:|:-------------------------:
<img src= "project1_yolo5/data/img/11752905643_a17ce5b925_c.jpg"  width="400">  |  <img src = 'project1_yolo5/data/img/23580321782_eec79c397f_c.jpg' width="400">

 tree:
 - flask_visual_result.ipynb - use for visual predict images(load data form 
        test_img_upload folder)
- viz_preprocessing.ipynb - same param for visual flask
- yolo5_parrot_toucan.ipynb - train and predict model

Result & Vizual:
```
Epoch   gpu_mem       box       obj       cls     total    labels  img_size     
149/149      5.8G   0.01731   0.01388  0.008292   0.03948        16       640: 100% 9/9 [00:02<00:00,  4.11it/s]   
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100% 3/3 [00:01<00:00,  2.28it/s]
                 all          38          38       0.979           1       0.995       0.909
              parrot          38          16       0.958           1       0.995       0.938
              toucan          38          22           1           1       0.995        0.88

150 epochs completed in 0.229 hours.
```
<img src = 'project1_yolo5/model_yolo/yolov5m_parrot8/test_batch0_pred.jpg'>
This how model make detection, but i not like magic and decided to make project2.

-------------------

## Project2_rcnn(fasterrcnn_resnet50)

Task to find where parrot on image:

  - New data, only parrot, image of parrots in the forest, maked resize(512*512), add tags(different size parrot on image)
  - model train:
      - train_test_split 40 epoch.
      - stratified folds by tags 40 epoch..
      - argumentation(img, bbox)
  - test data: images contain people and previously unseen compositions, image not changed(size).
  - for visual use streamlit

  Predict by each fold:
   <img src= "project2_rcnn/input/rcnn_predict.png"> 

  -v1. 3 may, result (no argumentation and scheduler):
  ```
  each folds [0.7906, 0.7209, 0.7843, 0.7850,0.715]    
  mean - 0.75916
  std  -  0.033770199880960146
  ```
  -v2. 5 may, result (with argumentation and scheduler):
  ```
  each folds [0.7576,0.7638,0.794,0.754, 0.7579] 
  meam - 0.76546
  std  - 0.01461199507254229
  ```
  Можно увидеть, что у первого варианта разброс значений значительно больше модель не стабильна, добавление аргументации и метода понижения скорости обучения привело к лучшим результатом(они далеки от идеальных), но мы уменьшили разброс, это означает что модель лучше обобщает данные результат стал стабильнее.

Update 6 may:

Мне стало интересно, как будет меняться результат если я добавлю новые данные. Я добавил где-то 50 картинок, но результат сильно не изменился даже, ухудшился. Поиски проблемы привели меня к дисбалансу(большие, средние и маленькие попугаи), как правило, когда снимают фотографии попугаев их хотят сделать, как можно крупнее поэтому изображений с маленькими попугаями мало.
Я на семплировал старые + новые в итоге выровнял дисбаланс.

Train folds_curve:
<img src= "project2_rcnn/input/folds_curve.png"> 
```
batch size = 4,
убрал A.VerticalFlip(p=0.5) у меня нет перевернутых попугаев
each folds [0.799,0.766,0.789,0.773,0.775]
mean - 0.7810114727800056,
std  - 0.011874766997491815
```
  
For visual use streamlit run src/streamlit_viz.py 

Loads pretrain model 1GB [link](https://drive.google.com/drive/folders/1zoVPg9hn-cKalaP8_5SqT6ocuHAeY9kt?usp=sharing)

## RESUME

По прошествии нескольких дней я вижу свои ошибки. Один из первых датасетов и разметок, я сделал для yolo, на этих данных модель смогла показать достойный результат.
Задача которую я ставил для Yolo это отличить попугай на изображении или тукан, с этой задачей модель справляется, но predicted box very big.

Проблема, это данные которые я сделал для yolo(они не приведены к одному формату, картинки разные по содержанию, размер объектов на изображении тоже очень разный, данных мало и тд).

Когда я стал делать rcnn я не смог на этих данных что-то на тренировать. Я сделал много вариантов датасетов и получил адекватные результаты(более тщательный выбор данных, один размер 512*512, добавил теги для фолдов, более четкая разметка, я понял какие данные нужно искать для лучшего прогноза).

Не спроста я не остановился на yolo, после rcnn я смог увидеть много проблем. Я оставлю, как есть данные для сравнения, но в ближайшие время переделаю их.

Update 6 may:

Как можно еще улучшить результат:
- добавить больше данных
- добавить аргументацию
- выбрать другую архитектуру
- выбрать другие оптимизаторы скорости обучения и т. д. 