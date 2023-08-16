#!/usr/bin/env python
# coding: utf-8

# ## Modelo ML - TFM

# ### RANDOM FOREST

# In[1]:


import sys

# Redirigir las advertencias a la salida estándar de errores (stderr)
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# In[2]:


# IMPORTO LIBRERIAS PRINCIPALES
import pandas as pd
import numpy as np
get_ipython().system('pip install sklearn')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


# In[3]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[4]:


# Visualizo los datos iniciales
data = pd.read_csv('ETL_chutes_ML.csv')
data = data.drop('Unnamed: 0', axis=1)
data.head(3)


# In[5]:


# Elimino columnas irrelevantes para el modelo
columnas_a_eliminar = ['duration', 'id', 'index', 'location', 'match_id', 
                       'possession_team_id', 'related_events', 
                       'shot_end_location', 'team', 'timestamp', 
                       'type', 'shot_end_location_z', 'related_events1', 
                       'related_events2','related_events3', 
                       'possession_team']
data=data.drop(columnas_a_eliminar, axis=1)


# In[6]:


# One hot encoding
encoded_data = pd.get_dummies(data, columns=['play_pattern', 'player', 
                                             'position', 
                                             'shot_body_part',
                                             'shot_technique', 
                                             'shot_type'])


# In[7]:


#Label encoder de la variable objetivo "shot_outcome"
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_data['shot_outcome'] = label_encoder.fit_transform(encoded_data['shot_outcome'])


# In[8]:


encoded_data.head(3)


# In[9]:


# Guardamos los datos codificados en un csv
encoded_data.to_csv('encoded_data.csv', index=False)


# In[10]:


# Creo 2 conjuntos, uno sin la variable objetivo y otro con la 
# variable objetivo
X = encoded_data.drop('shot_outcome', axis=1)  
y = encoded_data['shot_outcome'] 


# In[11]:


# Divido los datos en TRAIN y TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)


# In[12]:


# Creo el modelo y lo entreno
random_forest = RandomForestClassifier(n_estimators=50, 
                                       max_depth=None, 
                                       min_samples_split=2, 
                                       random_state=42)
random_forest.fit(X_train, y_train)


# In[13]:


# Hago las predicciones con el conjunto de test
y_pred_forest = random_forest.predict(X_test)
y_pred_forest


# In[14]:


# EXACTITUD del modelo RANDOM FOREST
accuracy_forest = accuracy_score(y_test, y_pred_forest)
print(f'Accuracy: {accuracy_forest:.3f}')


# In[15]:


from sklearn.metrics import confusion_matrix


# #### OPTIMIZACIÓN DEL MODELO RANDOM FOREST

# In[16]:


# MATRIZ DE CONFUSIÓN
confusion = confusion_matrix(y_test, y_pred_forest)
confusion


# In[17]:


from sklearn.model_selection import train_test_split, GridSearchCV

# Definir los hiperparámetros a explorar
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}


# In[18]:


# Crear el modelo de Random Forest
random_forest_opt = RandomForestClassifier(random_state=42)


# In[19]:


# Realizar la búsqueda de hiperparámetros con validación cruzada
grid_search = GridSearchCV(estimator=random_forest_opt, 
                           param_grid=param_grid, 
                           scoring='accuracy', 
                           cv=3)
grid_search.fit(X_train, y_train)


# In[20]:


# Obtener el mejor modelo con los mejores hiperparámetros
best_random_forest = grid_search.best_estimator_

# Predicciones en el conjunto de prueba utilizando el mejor modelo
y_pred_best_forest = best_random_forest.predict(X_test)

# Calcular la exactitud del modelo
accuracy_forest = accuracy_score(y_test, y_pred_best_forest)
print(f'Accuracy: {accuracy_forest:.3f}')

# Obtener los mejores hiperparámetros
best_params = grid_search.best_params_
print('Mejores hiperparámetros:', best_params)


# In[21]:


# Importancia de las variables en el modelo RANDOM FOREST
feature_importance1 = random_forest.feature_importances_
feature_importance1


# In[22]:


import matplotlib.pyplot as plt

# Obtener la importancia de las características desde el modelo
importances = random_forest.feature_importances_
feature_names = X_train.columns

# Gráfico de barras para mostrar la importancia de las características
plt.figure(figsize=(12, len(feature_names) * 0.3))  
plt.barh(range(len(feature_names)), importances, align='center')
plt.yticks(range(len(feature_names)), feature_names)

# Ajustar los márgenes para dar más espacio a las etiquetas del eje Y
plt.subplots_adjust(left=0.2)  # Ajusta este valor según sea necesario

plt.xlabel('Importancia de Características')
plt.title('Importancia de Características en el Modelo Random Forest')

plt.tight_layout()
plt.show()


# ## Prueba variables

# In[23]:


# Matriz de correlación entre las variables
correlation_matrix = encoded_data.corr()
correlation_matrix


# No vemos riesgo de multicolinealidad, aunque se va a tratar de eliminar algunas etiquetas con correlación próximas a 1 para evaluar como se modifica el modelo.

# In[24]:


encoded_data.head(2)


# In[25]:


# Prueba de eliminación de distintas variables
variables = ['shot_outcome', 'location_y']
variables_ = encoded_data[variables]


# In[26]:


# Creo 2 conjuntos, uno sin variables_ y otro con la variable objetivo
X = encoded_data.drop(variables_, axis=1)  
y = encoded_data['shot_outcome'] 


# In[27]:


# Divido los datos en TRAIN y TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)


# In[28]:


# Creo el modelo y lo entreno
random_forest = RandomForestClassifier(n_estimators=50, 
                                       max_depth=None, 
                                       min_samples_split=2, 
                                       random_state=42)
random_forest.fit(X_train, y_train)


# In[29]:


y_pred_forest = random_forest.predict(X_test)
y_pred_forest


# In[30]:


# EXACTITUD
accuracy_forest = accuracy_score(y_test, y_pred_forest)
print(f'Accuracy: {accuracy_forest:.3f}')


# ##### Resultados:
# - Si quitamos "minute" --> exactitud de 0.80
# - Si quitamos "player_id" --> exactitud de 0.765
# - Si quitamos "possession" --> exactitud de 0.812
# - Si quitamos "second" --> exactitud de 0.765
# - Si quitamos "shot_statsbomb_xg --> exactitud de 0.776
# ##### Si quitamos "location_y" --> exactitud de 0.835
# Podemos observar que las variables "location_x", "shot_end_location_x" y "shot_end_location_y" tienen gran peso en el modelo.

# ### Métricas FOREST

# In[31]:


# ACCURACY
accuracy_forest


# In[32]:


# R2
from sklearn.metrics import r2_score
r2_forest = r2_score(y_test, y_pred_forest)
r2_forest


# In[33]:


# F1-ScoreMacro
from sklearn.metrics import f1_score
f1_macro_forest = f1_score(y_test, y_pred_forest, average='macro')
f1_macro_forest


# In[34]:


# Realizo validación curzada para evaluar exhaustivamente la 
# exactitud del modelo TRAIN
from sklearn.model_selection import cross_val_score
cv_scores_rf_train = cross_val_score(random_forest, X_train, y_train, 
                                     cv=5, 
                                     scoring='accuracy')
cv_scores_rf_train


# In[35]:


# Realizo validación curzada para evaluar exhaustivamente la 
# exactitud del modelo TEST
from sklearn.model_selection import cross_val_score
cv_scores_rf_test = cross_val_score(random_forest, X_test, y_test, 
                                    cv=5, 
                                    scoring='accuracy')
cv_scores_rf_test


# In[36]:


# MATRIZ DE CONFUSIÓN
from sklearn.metrics import confusion_matrix
confusion_forest = confusion_matrix(y_test, y_pred_forest)
confusion_forest


# In[37]:


# Reporte de Clasificación
class_report_forest = classification_report(y_test, y_pred_forest)
for line in class_report_forest.split('\n'):
    print(line)


# In[38]:


# Curva ROC - AUC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Probabilidades de predicción para cada clase en el conjunto de prueba
y_probs_rf = random_forest.predict_proba(X_test)

# Calcular la curva ROC y el área bajo la curva (AUC) para cada clase
plt.figure(figsize=(8, 6))
for i in range(len(random_forest.classes_)):
    fpr, tpr, _ = roc_curve(y_test, y_probs_rf[:, i], 
                            pos_label=random_forest.classes_[i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, 
             label=f'Clase {random_forest.classes_[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC por Clase (Random Forest)')
plt.legend(loc='lower right')
plt.show()


# ## XGBoost

# In[39]:


# IMPORTO LIBRERIAS NECESARIAS PARA EL MODELO
get_ipython().system('pip install xgboost')
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# In[40]:


# Divido los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)


# In[41]:


# Creo y entreno el modelo XGBoost
xgboost_model = xgb.XGBClassifier(
    n_estimators=50,  # Número de árboles
    max_depth=3,       # Profundidad máxima de los árboles
    learning_rate=0.1, # Tasa de aprendizaje
    random_state=42
)
xgboost_model.fit(X_train, y_train)


# In[42]:


# Realizo predicciones en el conjunto de prueba
y_pred_XG = xgboost_model.predict(X_test)
y_pred_XG


# In[43]:


# Calculo la exactitud del modelo
accuracy_XG = accuracy_score(y_test, y_pred_XG)

# Mostrar la exactitud del modelo
print(f'Exactitud del modelo XGBoost: {accuracy_XG:.3f}')


# #### OPTIMIZACIÓN DEL MODELO XGBoost

# In[44]:


# Definir la cuadrícula de hiperparámetros a explorar
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001]
}


# In[45]:


# Realizar Grid Search
grid_search = GridSearchCV(estimator=xgboost_model, 
                           param_grid=param_grid, 
                           scoring='accuracy', 
                           cv=3)
grid_search.fit(X_train, y_train)


# In[46]:


# Obtener el mejor modelo y sus hiperparámetros
best_xgboost_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_params


# In[47]:


# Realizar predicciones en el conjunto de prueba con el mejor modelo
y_pred_XG = best_xgboost_model.predict(X_test)


# In[48]:


# Calcular la exactitud del modelo
accuracy_XG = accuracy_score(y_test, y_pred_XG)

# Mostrar la exactitud y los mejores hiperparámetros
print(f'Exactitud del modelo XGBoost optimizado: {accuracy_XG:.2f}')
print('Mejores hiperparámetros:', best_params)


# In[49]:


feature_importance2 = xgboost_model.feature_importances_
feature_importance2


# ### Métricas XGBoost

# In[50]:


# ACCURACY
accuracy_XG


# In[51]:


# R2
from sklearn.metrics import r2_score
r2_XG = r2_score(y_test, y_pred_XG)
r2_XG


# In[52]:


# F1-ScoreMacro
from sklearn.metrics import f1_score
f1_macro_XG = f1_score(y_test, y_pred_XG, average='macro')
f1_macro_XG


# In[53]:


# Realizo validación curzada para evaluar exhaustivamente 
# la exactitud del modelo TRAIN
from sklearn.model_selection import cross_val_score
cv_scores_XG_train = cross_val_score(xgboost_model, X_train, y_train, 
                                     cv=5, 
                                     scoring='accuracy')
cv_scores_XG_train


# In[54]:


# Realizo validación curzada para evaluar exhaustivamente la 
# exactitud del modelo TEST
from sklearn.model_selection import cross_val_score
cv_scores_XG_test = cross_val_score(xgboost_model, X_test, y_test, 
                                    cv=5, 
                                    scoring='accuracy')
cv_scores_XG_test


# In[55]:


# MATRIZ DE CONFUSIÓN
confusion_XG = confusion_matrix(y_test, y_pred_XG)
confusion_XG


# In[56]:


# Reporte de Clasificación
class_report_XG = classification_report(y_test, y_pred_XG)
for line in class_report_XG.split('\n'):
    print(line)


# In[57]:


# Curva ROC - AUC 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Probabilidades de predicción para cada clase en el conjunto de prueba
y_probs_xg = xgboost_model.predict_proba(X_test)

# Calcular la curva ROC y el área bajo la curva (AUC) para cada clase
plt.figure(figsize=(8, 6))
for i in range(len(xgboost_model.classes_)):
    fpr, tpr, _ = roc_curve(y_test, y_probs_xg[:, i], 
                            pos_label=xgboost_model.classes_[i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, 
             label=f'Clase {xgboost_model.classes_[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC por Clase (XGBoost)')
plt.legend(loc='lower right')
plt.show()


# ## NAIVE BAYES

# In[58]:


from sklearn.naive_bayes import MultinomialNB


# In[59]:


# Crear y entrenar el modelo Naive Bayes
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train, y_train)


# In[60]:


# Realizar predicciones en el conjunto de prueba
y_pred_bayes = naive_bayes_model.predict(X_test)


# In[61]:


# Calcular la exactitud del modelo
accuracy_bayes = accuracy_score(y_test, y_pred_bayes)

# Mostrar la exactitud del modelo
print(f'Exactitud del modelo Naive Bayes: {accuracy_bayes:.2f}')


# ### Métricas

# In[62]:


# ACCURACY
accuracy_bayes


# In[63]:


# R2
from sklearn.metrics import r2_score
r2_bayes = r2_score(y_test, y_pred_bayes)
r2_bayes


# In[64]:


# F1-ScoreMacro
from sklearn.metrics import f1_score
f1_macro_bayes = f1_score(y_test, y_pred_bayes, average='macro')
f1_macro_bayes


# In[65]:


# Realizo validación curzada para evaluar exhaustivamente 
# la exactitud del modelo TRAIN
from sklearn.model_selection import cross_val_score
cv_scores_bayes_train = cross_val_score(naive_bayes_model, X_train, 
                                        y_train, 
                                        cv=5, 
                                        scoring='accuracy')
cv_scores_bayes_train


# In[66]:


# Realizo validación curzada para evaluar exhaustivamente la 
# exactitud del modelo TEST
from sklearn.model_selection import cross_val_score
cv_scores_bayes_test = cross_val_score(naive_bayes_model, X_test, 
                                       y_test, 
                                       cv=5, 
                                       scoring='accuracy')
cv_scores_bayes_test


# In[67]:


# MATRIZ DE CONFUSIÓN
confusion_bayes = confusion_matrix(y_test, y_pred_bayes)
confusion_bayes


# In[68]:


# Reporte de Clasificación
class_report_bayes = classification_report(y_test, y_pred_bayes)
for line in class_report_bayes.split('\n'):
    print(line)


# In[69]:


# Curva ROC - AUC 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Probabilidades de predicción para cada clase en el conjunto de prueba
y_probs_nb = naive_bayes_model.predict_proba(X_test)

# Calcular la curva ROC y el área bajo la curva (AUC) para cada clase
plt.figure(figsize=(8, 6))
for i in range(len(naive_bayes_model.classes_)):
    fpr, tpr, _ = roc_curve(y_test, y_probs_nb[:, i], 
                            pos_label=naive_bayes_model.classes_[i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, 
             label=f'Clase {naive_bayes_model.classes_[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC por Clase (Naive Bayes)')
plt.legend(loc='lower right')
plt.show()


# ## SVM

# In[70]:


from sklearn.svm import SVC


# In[71]:


# Crear y entrenar el modelo SVM
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred_svm = svm_model.predict(X_test)


# In[72]:


# Calcular la exactitud del modelo
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Mostrar la exactitud del modelo
print(f'Exactitud del modelo SVM: {accuracy_svm:.3f}')


# ### Métricas SVM

# In[73]:


# ACCURACY
accuracy_svm


# In[74]:


# R2
from sklearn.metrics import r2_score
r2_svm = r2_score(y_test, y_pred_svm)
r2_svm


# In[75]:


# F1-ScoreMacro
from sklearn.metrics import f1_score
f1_macro_svm = f1_score(y_test, y_pred_svm, average='macro')
f1_macro_svm


# In[76]:


# Realizo validación curzada para evaluar exhaustivamente la 
# exactitud del modelo TRAIN
from sklearn.model_selection import cross_val_score
cv_scores_svm_train = cross_val_score(svm_model, X_train, y_train, 
                                      cv=5, 
                                      scoring='accuracy')
cv_scores_svm_train


# In[77]:


# Realizo validación curzada para evaluar exhaustivamente la 
# exactitud del modelo TEST
from sklearn.model_selection import cross_val_score
cv_scores_svm_test = cross_val_score(svm_model, X_test, y_test, 
                                     cv=5, 
                                     scoring='accuracy')
cv_scores_svm_test


# In[78]:


# MATRIZ DE CONFUSIÓN
confusion_svm = confusion_matrix(y_test, y_pred_svm)
confusion_svm


# In[79]:


# Reporte de Clasificación
class_report_svm = classification_report(y_test, y_pred_svm)
for line in class_report_svm.split('\n'):
    print(line)


# ## GRADIENT BOOSTING

# In[80]:


from sklearn.ensemble import GradientBoostingClassifier

# Crear y entrenar el modelo Gradient Boosting
gradient_boosting_model = GradientBoostingClassifier(random_state=42)
gradient_boosting_model.fit(X_train, y_train)


# In[81]:


# Realizar predicciones en el conjunto de prueba
y_pred_gb = gradient_boosting_model.predict(X_test)

# Calcular la exactitud del modelo
accuracy_gb = accuracy_score(y_test, y_pred_gb)

# Mostrar la exactitud del modelo
print(f'Exactitud del modelo Gradient Boosting: {accuracy_gb}')


# #### OPTIMIZACIÓN DEL MODELO GB

# In[82]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Definir los hiperparámetros a explorar
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001]
}


# In[83]:


# Realizar la búsqueda de hiperparámetros con validación cruzada
grid_search = GridSearchCV(gradient_boosting_model, param_grid, 
                           cv=5, 
                           scoring='accuracy')
grid_search.fit(X_train, y_train)


# In[84]:


# Obtener el mejor modelo con los mejores hiperparámetros
best_gradient_boosting = grid_search.best_estimator_

# Predicciones en el conjunto de prueba utilizando el mejor modelo
y_pred_best_gb = best_gradient_boosting.predict(X_test)

# Calcular la exactitud del modelo
accuracy_gb_opt = accuracy_score(y_test, y_pred_best_gb)
print(f'Accuracy: {accuracy_gb:.1f}')

# Obtener los mejores hiperparámetros
best_params_gb = grid_search.best_params_
print('Mejores hiperparámetros:', best_params_gb)


# In[85]:


feature_importance3 = gradient_boosting_model.feature_importances_
feature_importance3


# ### Métricas GB

# In[86]:


# ACCURACY
accuracy_gb


# In[87]:


# R2
from sklearn.metrics import r2_score
r2_gb = r2_score(y_test, y_pred_gb)
r2_gb


# In[88]:


# F1-ScoreMacro
from sklearn.metrics import f1_score
f1_macro_gb = f1_score(y_test, y_pred_gb, average='macro')
f1_macro_gb


# In[89]:


# Realizo validación curzada para evaluar exhaustivamente la 
# exactitud del modelo TRAIN
from sklearn.model_selection import cross_val_score
cv_scores_gb_train = cross_val_score(gradient_boosting_model, X_train, 
                                     y_train, 
                                     cv=5, 
                                     scoring='accuracy')
cv_scores_gb_train


# In[90]:


# Realizo validación curzada para evaluar exhaustivamente la 
# exactitud del modelo TEST
from sklearn.model_selection import cross_val_score
cv_scores_gb_test = cross_val_score(gradient_boosting_model, X_test, 
                                    y_test, 
                                    cv=5, 
                                    scoring='accuracy')
cv_scores_gb_test


# In[91]:


# MATRIZ DE CONFUSIÓN
confusion_gb = confusion_matrix(y_test, y_pred_gb)
confusion_gb


# In[92]:


# Reporte de Clasificación
class_report_gb = classification_report(y_test, y_pred_gb)
for line in class_report_gb.split('\n'):
    print(line)


# In[93]:


# Curva ROC-AUC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Probabilidades de predicción para cada clase en el conjunto de prueba
y_probs_gb = gradient_boosting_model.predict_proba(X_test)

# Calcular la curva ROC y el área bajo la curva (AUC) para cada clase
plt.figure(figsize=(8, 6))
for i in range(len(gradient_boosting_model.classes_)):
    fpr, tpr, _ = roc_curve(y_test, 
                            y_probs_gb[:, i], 
                            pos_label=gradient_boosting_model.classes_[i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, 
             label=f'Clase {gradient_boosting_model.classes_[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC por Clase (Gradient Boosting)')
plt.legend(loc='lower right')
plt.show()


# - Eliminando la variable "location_y" todos los algoritmos aumentan su exactitud.

# In[94]:


## Output de las validaciones cruzadas
print (f'RANDOM FOREST')
print(f'Validación cruzada de RF TRAIN: {cv_scores_rf_train}')
print(f'Validación cruzada de RF TEST: {cv_scores_rf_test}')
print(f' ')
print (f'XGBoost')
print(f'Validación cruzada de XG TRAIN: {cv_scores_XG_train}')
print(f'Validación cruzada de XG TEST: {cv_scores_XG_test}')
print(f' ')
print (f'NAIVE BAYES')
print(f'Validación cruzada de NB TRAIN: {cv_scores_bayes_train}')
print(f'Validación cruzada de NB TEST: {cv_scores_bayes_test}')
print(f' ')
print (f'SVM')
print(f'Validación cruzada de SVM TRAIN: {cv_scores_svm_train}')
print(f'Validación cruzada de SVM TEST: {cv_scores_svm_test}')
print(f' ')
print (f'GRADIENT BOOSTING')
print(f'Validación cruzada de GB TEST: {cv_scores_gb_train}')
print(f'Validación cruzada de GB TRAIN: {cv_scores_gb_test}')

