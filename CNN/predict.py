from keras.models import load_model

# Loading and compiling presaved trained CNN
model = load_model('safedriving_classification.h5')
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Prediction
from keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory("dataset/test_set",
                                                    target_size = (64, 64),
                                                    color_mode = "rgb",
                                                    shuffle = False,
                                                    class_mode = 'categorical',
                                                    batch_size = 1)
filenames = test_generator.filenames
nb_samples = len(filenames)
predict = model.predict_generator(test_generator,steps = nb_samples)


# Giving class label from probabilities
from keras.utils import np_utils
import numpy as np

def probas_to_classes(y_pred):
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        return categorical_probas_to_classes(y_pred)
    return np.array([1 if p > 0.5 else 0 for p in y_pred])

def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)

 
y_classes = probas_to_classes(predict)


# Calculating confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score

y_true = np.array([0] * 350 + [1] * 318 + [2] * 325 + [3] * 329 + [4] * 327 + [5] * 325 + [6] * 326 + [7] * 281 + [8] * 268 + [9] * 299)
#y_pred = predict > 0.5

cm = confusion_matrix(y_true, y_classes)
ac = accuracy_score(y_true, y_classes)
