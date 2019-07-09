
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import isfile

import keras
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from sklearn.metrics.classification import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
from dbn.tensorflow import SupervisedDBNClassification as DBNC
from dbn.utils import to_categorical

#%% [markdown]
# ##### 3.2.2 Preprocesamiento de imágenes
# 
# Esta primera parte es para demostrar el como se manejaron las imágenes y como entrenó el modelo, pero luego usaremos un modelo ya entrenado para mostrar los resultados.

#%%
pti = 'images/train/' # Path To Images
ext = '.jpg' # Image extension

# Cargar datos sobre archivos en pd Dataframe
traindata = pd.read_csv("train.csv") 

# Agregar a los nombres de los archivos los paths y extensiones correspondientes
traindata['filename'] = traindata['filename'].apply(lambda x: pti + x + ext)

traindata.shape


#%%
# Based on: https://www.analyticsvidhya.com/blog/2019/01/build-image-classification-model-10-minutes/

# Creación de vector de inputs al modelo
train_images = []

color = "grayscale" # Se elige la escala de grises para mantener simple el modelo
img_size = (96, 128) # Tamaño de pixeles a considerar 3:4 ratio (alto:ancho)
img_ratio = 0.05 # 5% de las imagenes serán usadas, importante para no consumir toda la ram

for i in tqdm(range(int(traindata.shape[0] * img_ratio))):
    
    # Cargamos la imagen correspondiente en blanco y negro, y con el tamaño deseado
    img = image.load_img(traindata['filename'][i], target_size=img_size, color_mode=color) # El cambiar el graysacale hace que el shape tenga profundidad 3
    # pasamos la imagen a una matriz de grayscale pixels
    img = image.img_to_array(img)
    # Escalamos el valor de cada pixel (255 -> 1)
    img = img/255
    # Flatten por que la librería utilizada recibe inputs de una dimensión
    train_images.append(img.flatten())

# Transformamos nuestro vector de inputs en un np array
X = np.array(train_images) 


#%%
# Ejemplo de imagen, y su vector obtenido
img = image.load_img(traindata['filename'][np.random.randint(0, int(traindata.shape[0] * img_ratio))], target_size=(96,128), color_mode="grayscale")
print(image.img_to_array(img).flatten()/255)
img


#%%
# Definición de vector de objetivos para el vector de inputs obtenido anteriormente

targets = traindata["landmark_id"][:int(traindata.shape[0] * img_ratio)]


#%%
# Split de datos

X_train, X_test, y_train, y_test = train_test_split(X, targets, random_state=42, test_size=0.2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

#%% [markdown]
# ##### 3.2.3 Entrenamiento del modelo

#%%
# # Training
# classifier = DBNC(hidden_layers_structure=[512, 256, 256],
#                   learning_rate_rbm=0.05, # learning rate para RBM pretraining
#                   learning_rate=0.1, # Learning rate para el backpropagation de los nodos
#                   n_epochs_rbm=10, # Epochs para RBM pretraining (n epochs x capa)
#                   n_iter_backprop=100, # Packpropagation iterations
#                   batch_size=46, # Tamaño de batch
#                   activation_function='sigmoid', # Función de activación
#                   dropout_p=0.1) # Dropout de nodos para evitar overfitting

# classifier.fit(X_train, y_train) # Entrenamiento del modelo según datos procesados en 3.2.2

#%% [markdown]
# ##### 3.2.4 Obtención de  resultados con modelo ya entrenado

#%%
# En primer lugar, cargamos los datos
tclassifier = DBNC.load('model.pkl')

# Datos trained classifier
tclassifier, tclassifier.unsupervised_dbn.hidden_layers_structure


#%%
# Predicción de test
y_pred = tclassifier.predict(X_test)


#%%
# Análisis de resultados
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#%% [markdown]
# ##### 3.2.5 Análisis de resultados
# 
# Se obserba una precisión 48%, lo que significa una mejora significante sobre random para 13 clases (~7,7%), por lo que podemos decir que nuestra modelación logró aprender en alguna medida caracterísiticas de ciertos landmarks. Esto se ve reflejado especialemente en el landmark de *id* 5554, el cual tuvo una precisión de 71% al momento de su clasificación. Esto puede ser explicado con que, como se ve a continuación, este landmark corresponde a las Torres Petronas, las cuales están rodeadas de pocas edificaciones u objetos que podrían entorpecer su detección. También, debido a su forma, este landmark es fotografiado casi siempre de ángulos similares, haciendo su detección un poco mas trivial (usando una definición muy amplia de trivial).  

#%%
img1 = image.load_img(traindata['filename'][traindata.landmark_id == 5554].iloc[0], target_size=(480,640))
img2 = image.load_img(traindata['filename'][traindata.landmark_id == 5554].iloc[1], target_size=(480,640))
img3 = image.load_img(traindata['filename'][traindata.landmark_id == 5554].iloc[2], target_size=(480,640))
img4 = image.load_img(traindata['filename'][traindata.landmark_id == 5554].iloc[3], target_size=(480,640))

plt.imshow(img1)
plt.show()
plt.imshow(img2)
plt.show()
plt.imshow(img3)
plt.show()
plt.imshow(img4)
plt.show()

#%% [markdown]
# Uno de los *landmarks* que peor resultados mostro, por otro lado corresponde a un edificio cuyas fotos se puede observar no son muy descriptivas del mismo, de hecho es dificil establecer a simple vista una relación entre las primeras 4 fotos encontradas:

#%%
img1 = image.load_img(traindata['filename'][traindata.landmark_id == 10900].iloc[0], target_size=(480,640))
img2 = image.load_img(traindata['filename'][traindata.landmark_id == 10900].iloc[1], target_size=(480,640))
img3 = image.load_img(traindata['filename'][traindata.landmark_id == 10900].iloc[2], target_size=(480,640))
img4 = image.load_img(traindata['filename'][traindata.landmark_id == 10900].iloc[3], target_size=(480,640))

plt.imshow(img1)
plt.show()
plt.imshow(img2)
plt.show()
plt.imshow(img3)
plt.show()
plt.imshow(img4)
plt.show()

#%% [markdown]
# ## 4. Conclusión
# 
# Si bien los resultados obtenidos no son tan buenos como se pensó, se logró obtener una clasificación relativamente buena de algunos *landmarks*, como fue en el caso de las Torres Petronas. Esto demuestra que DBN puede ser utilizado para la clasificación de landmarks, pero es necesario aplicar varios ajustes y mejoras para disminuir errores.
# 
# Tal vez una de las mayores fuentes de error en nuestro modelamiento corresponde a el asumir el input como un vector de valores, en vez de una matriz, pero esto corresponde a una variable innevitable dada la implementación disponible para el problema. De ser posible ingresar la imagen como matriz, en vez de un solo vector, se podría extraer aún más features de las imágenes, al obtener no solo relaciones horizontales entre los pixeles, sino que además una relacion espacial de dos dimensiones entre conjuntos de pixeles vecinos. Y en un caso ideal, considerar además el utilizar los colores de las imágenes (*landmarks*) como otro canal caracterísitico de ellos, ya que si bien dos edificios pueden ser muy similares, pueden diferir en sus colores y establecer ahí una diferencia. 
# 
# Otro gran obstáculo al momento de resolver este problema fue la capacidad de cómuputo disponible, ya que en el caso de querer usar más imágenes para entraining, o agregar más layers, o aumentar el batch size, se consumía toda la RAM disponible en el computador, botando la ejecución. Se intentó trabajar en google colab para enfrentar esto, pero tampoco fue suficiente. En caso de poder aumentar la capacidad de computo sería además posible aplicar ambos puntos expresados en el párrafo anterior (características espaciales y colores), permitiendo seguramente una mejor clasificación
# 
# Para tener en consideración en trabajos futuros sobre el tema de clasificación de imágenes con DBN, resulta escencial considerar la opción de una mezcla de modelos, donde uno corresponde a una red con capas convolucionales para detectar distintos patrones en la imagen, y que luego ésta se conecte a una DBN para el análisis y especificación más profunda de los atributos encontrados en la capa anterior.
#%% [markdown]
# ## 5. Referencias - Bibliografía
# 
# **[ 1 ]**  *Google Landmark Recognition Challenge* https://www.kaggle.com/c/landmark-recognition-challenge  
# **[ 2 ]**  Bishop, C. M. (2006). *Pattern recognition and machine learning*. Springer.  
# **[ 3 ]**  Koller, D., & Friedman, N. (2009). *Probabilistic graphical models: principles and techniques*. MIT press.  
# **[ 4 ]**  Fischer, A., & Igel, C. (2012, September). *An introduction to restricted Boltzmann machines. In Iberoamerican Congress on Pattern Recognition* (pp. 14-36). Springer, Berlin, Heidelberg.   
# **[ 5 ]**  Montúfar, G. (2016, June). *Restricted Boltzmann Machines: Introduction and Review. In Information Geometry and its Applications IV* (pp. 75-115). Springer, Cham.  
# **[ 6 ]**  G.E. Hinton and R.R. Salakhutdinov, *Reducing the Dimensionality of Data with Neural Networks*, Science, 28 July 2006, Vol. 313. no. 5786, pp. 504 - 507.
# **[ 7 ]**  	G.E. Hinton, S. Osindero, and Y. Teh, “A fast learning algorithm for deep belief nets”, Neural Computation, vol 18, 2006   
# **[ 8 ]**  [A Python implementation of Deep Belief Networks built upon NumPy and TensorFlow with scikit-learn compatibility](https://github.com/albertbup/deep-belief-network)    
# **[ 9 ]**  
# **[ 10 ]** 

#%%



