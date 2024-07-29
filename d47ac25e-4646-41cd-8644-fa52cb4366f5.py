#!/usr/bin/env python
# coding: utf-8

# ¡Hola Jenniffer! Como te va?
# 
# Mi nombre es Facundo Lozano! Un gusto conocerte, seré tu revisor en este proyecto.
# 
# A continuación un poco sobre la modalidad de revisión que usaremos:
# 
# Cuando enccuentro un error por primera vez, simplemente lo señalaré, te dejaré encontrarlo y arreglarlo tú cuenta. Además, a lo largo del texto iré haciendo algunas observaciones sobre mejora en tu código y también haré comentarios sobre tus percepciones sobre el tema. Pero si aún no puedes realizar esta tarea, te daré una pista más precisa en la próxima iteración y también algunos ejemplos prácticos. Estaré abierto a comentarios y discusiones sobre el tema.
# 
# Encontrará mis comentarios a continuación: **no los mueva, modifique ni elimine**.
# 
# Puedes encontrar mis comentarios en cuadros verdes, amarillos o rojos como este:
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Exito. Todo se ha hecho de forma exitosa.
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Observación. Algunas recomendaciones.
# </div>
# 
# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Necesita arreglos. Este apartado necesita algunas correcciones. El trabajo no puede ser aceptado con comentarios rojos. 
# </div>
# 
# Puedes responder utilizando esto:
# 
# <div class="alert alert-block alert-info">
# <b>Respuesta de estudiante.</b> <a class="tocSkip"></a>
# </div> Hola, Facundo, muchas gracias por tus comentarios en mi proyecto, ayuda mucho a ir aprendiendo y perfeccionando las habilidades aprendidas en TripleTen. Agradecida con tus buenas vibras para continuar aprendidendo y teniendo éxitos en mis estudios. Saludos.

# # Introducción:
# Megaline es una empresa de telecomunicaciones que ha observado que una gran parte de sus clientes sigue utilizando planes heredados que no optimizan sus necesidades actuales. Con el objetivo de mejorar la experiencia del cliente y maximizar la eficiencia de sus servicios, Megaline ha lanzado nuevos planes de suscripción: Smart y Ultra. Este estudio no solo contribuirá a mejorar la satisfacción del cliente al ofrecerles planes más adecuados a sus necesidades, sino que también ayudará a Megaline a optimizar la gestión de sus recursos y a diseñar estrategias más efectivas para la retención de clientes.
# 
# # Objetivo:
# El propósito de este proyecto es desarrollar un modelo de machine learning que analice el comportamiento de los clientes de Megaline y recomiende el plan más adecuado (Smart o Ultra) con la mayor precisión posible. Para lograr este objetivo, contamos con un conjunto de datos que contiene información detallada sobre el uso mensual de servicios por parte de los suscriptores que ya han migrado a los nuevos planes.
# 
# La metodología seguida en este proyecto incluye los siguientes pasos:
# 
# 1. Carga y exploración de datos: Leer y examinar el archivo de datos proporcionado para entender su estructura y contenido.
# 2. Segmentación de datos: Dividir el conjunto de datos en subconjuntos de entrenamiento, validación y prueba para garantizar una evaluación imparcial del modelo.
# 3. Entrenamiento y evaluación de modelos: Probar varios algoritmos de clasificación, ajustar hiperparámetros y evaluar su rendimiento para identificar el modelo más preciso.
# 4. Selección del mejor modelo: Evaluar el mejor modelo en el conjunto de prueba para obtener una medida precisa de su exactitud.
# 5. Prueba de cordura: Realizar pruebas adicionales para asegurar la validez y robustez del modelo desarrollado.

# <div class="alert alert-block alert-success">
# <b>Review General. (Iteración 1) </b> <a class="tocSkip"></a>
# 
# Buenas Jenniffer! Siempre me tomo este tiempo al inicio de tu proyecto para comentarte mis apreciaciones generales de esta iteración de tu entrega. 
# 
# Me gusta comenzar dando la bienvenida al mundo de los datos a los estudiantes, te deseo lo mejor y espero que consigas lograr tus objetivos. Personalmente me gusta brindar el siguiente consejo, "Está bien equivocarse, es normal y es lo mejor que te puede pasar. Aprendemos de los errores y eso te hará mejor programando ya que podrás descubrir cosas a medida que avances y son estas cosas las que te darán esa experiencia para ser mejor como  Data Scientist"
# 
# Ahora si yendo a esta notebook. Quería felicitarte Jennifer porque has logrado resolver todos los pasos implementando grandes lógicas, se ha notado tu manejo sobre python y las herramientas ML utilizadas. Muy bien hecho! Solo hemos tenido un detalle pero que seguro nos tomara un momento para resolverlo, te he dejado un comentario con el contexto necesario para que puedas resolverlo.
# 
# Espero con ansias a nuestra próxima iteración Jennifer, exitos y saludos!

# In[2]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier


# In[4]:


data = pd.read_csv('/datasets/users_behavior.csv')


# In[5]:


print(data.head())


# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)
#     
# </b> <a class="tocSkip"></a>
# 
#     
# Excelente implementación de importaciones y carga de datos. Felicitaciones por mantener los procesos en celdas separadas! A la vez excelente implementación de los métodos para observar la composición de los datos!

# In[6]:


# Segmentar los datos en entrenamiento (60%), validación (20%) y prueba (20%)
data_train, data_temp = train_test_split(data, test_size=0.4, random_state=12345)
data_valid, data_test = train_test_split(data_temp, test_size=0.5, random_state=12345)


# In[7]:


train_size = len(data_train)
val_size = len(data_valid)
test_size = len(data_test)
print(train_size, val_size, test_size)


# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)
#     
# </b> <a class="tocSkip"></a>
# 
#     
# Excelente división de los datos en los 3 conjuntos correpsondientes Jennifer, excelente implementación de train_test_split()!

# Se dividieron los datos en conjuntos de entrenamiento, validación y prueba. Una división común de 60% para entrenamiento, 20% para validación y 20% para prueba. Esta estrategia asegura que el modelo se entrene con un conjunto de datos suficientemente grande, mientras que se utiliza un conjunto separado para ajustar hiperparámetros y evaluar el rendimiento del modelo de manera imparcial.

# In[8]:


# Definir características y etiquetas
features = ['calls', 'minutes', 'messages', 'mb_used']
target = data['is_ultra']

features_train = data_train[features]
target_train = data_train['is_ultra']
features_val = data_valid[features]
target_val = data_valid['is_ultra']

# Inicializar modelos
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
}

# Entrenar y evaluar modelos
best_model = None
best_accuracy = 0
model_performance = {}

for model_name, model in models.items():
    model.fit(features_train, target_train)
    predictions = model.predict(features_val)
    accuracy = accuracy_score(target_val, predictions)
    model_performance[model_name] = accuracy
    print(f'{model_name} Accuracy: {accuracy}')
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print(f'Mejor modelo: {best_model}')
print(f'Mejor exactitud: {best_accuracy}')


# In[12]:


# Definir los hiperparámetros a probar
n_estimators_options = [50, 100, 150]
max_depth_options = [None, 10, 20, 30]

# Inicializar variables para almacenar el mejor modelo y su precisión
best_model = None
best_accuracy = 0
best_params = {}

# Probar diferentes combinaciones de hiperparámetros
for n_estimators in n_estimators_options:
    for max_depth in max_depth_options:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=12345)
        model.fit(features_train, target_train)
        predictions = model.predict(features_val)
        accuracy = accuracy_score(target_val, predictions)
                
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
            }

print(f'Mejor modelo: {best_model}')
print(f'Mejor combinación de hiperparámetros: {best_params}')
print(f'Exactitud del mejor modelo en el conjunto de validación: {best_accuracy}')

# Evaluar el mejor modelo en el conjunto de prueba
features_test = data_test[features]
target_test = data_test['is_ultra']
test_predictions = best_model.predict(features_test)
test_accuracy = accuracy_score(target_test, test_predictions)

print(f'Exactitud en el conjunto de prueba: {test_accuracy}')


# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)
#     
# </b> <a class="tocSkip"></a>
# 
#     
# Procedimiento perfecto Jennifer! En este caso guardando diferentes mmodelos basicos para luego iterar, entrenarlos y evaluarlos con las metricas correspondientes, muy bien hecho!

# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor. (Iteración 1)
#     
# </b> <a class="tocSkip"></a>
# 
#     
# Aquí deberiamos realizar un agregado Jennifer, es muy importante más alla de probar modelos basicos de probar modelos con diferentes hiperparametros para observar como estos impactan de forma positiva. Te invito a que agregues varios hiperparametros en al menos uno modelo.

# El modelo con el mejor rendimiento es el Random Forest, con una exactitud del 79,62%.

# In[11]:


# Evaluar el mejor modelo en el conjunto de prueba
features_test = data_test[features]
target_test = data_test['is_ultra']
test_pred = best_model.predict(features_test)
test_accuracy = accuracy_score(target_test, test_pred)

print(f'Exactitud en el conjunto de prueba: {test_accuracy}')


# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)
#     
# </b> <a class="tocSkip"></a>
# 
#     
# Y una excelente prueba del mejor modelo contra el tercer conjunto, bien hecho!

# El modelo Random Forest fue evaluado en el conjunto de prueba y obtuvo una exactitud de 78.38%, superando el umbral establecido del 75%.

# In[8]:


# Comparar con un modelo Dummy
dummy_model = DummyClassifier(strategy='most_frequent')
dummy_model.fit(features_train, target_train)
dummy_pred = dummy_model.predict(features_val)
dummy_accuracy = accuracy_score(target_val, dummy_pred)

print(f'Exactitud del modelo Dummy: {dummy_accuracy}')


# <div class="alert alert-block alert-success">
# 
# <b>Comentario del revisor. (Iteración 1)
#     
# </b> <a class="tocSkip"></a>
# 
#     
# Excelente decision de implementacion del DummyClassifier() para nuestro modelo de cordura para este caso Jennifer, implementación perfecta!

# Se realizó una prueba de cordura utilizando un modelo Dummy que predice siempre la clase más frecuente.
# 
# La exactitud del modelo Random Forest (78.38%) fue significativamente superior a la del modelo Dummy, demostrando que el modelo Random Forest tiene una capacidad predictiva real y no está simplemente sobreajustando los datos.

# # Conclusiones
# - Modelo Elegido: El modelo Random Forest fue seleccionado debido a su alto rendimiento.
# - Desempeño: Con una exactitud de 78.38% en el conjunto de prueba, el modelo supera el umbral requerido y se considera apto para recomendar planes de Megaline.
# - **Recomendación:** Implementar el modelo Random Forest en el entorno de producción de Megaline para mejorar la recomendación de planes y potencialmente aumentar la satisfacción y retención de clientes.
