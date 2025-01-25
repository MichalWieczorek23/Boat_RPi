from __future__ import print_function
from keras import layers
from keras import models

# Wczytanie danyc

train_dir = 'mixed_train'
val_dir = 'mixed_val'
test_dir = 'test'
'''
# Katalogi poszczegolnych klas
train_butelka = os.path.join(train_dir, 'ButelkaPlast')
val_butelka = os.path.join(val_dir, 'ButelkaPlast')
test_butelka = os.path.join(test_dir, 'ButelkaPlast')

train_czipsy = os.path.join(train_dir, 'CzipsyPaka')
val_czipsy = os.path.join(val_dir, 'CzipsyPaka')
test_czipsy = os.path.join(test_dir, 'CzipsyPaka')

train_kapsle = os.path.join(train_dir, 'Kapsle')
val_kapsle = os.path.join(val_dir, 'Kapsle')
test_kapsle = os.path.join(test_dir, 'Kapsle')

train_kubeczek = os.path.join(train_dir, 'KubeczekPlast')
val_kubeczek = os.path.join(val_dir, 'KubeczekPlast')
test_kubeczek = os.path.join(test_dir, 'KubeczekPlast')

train_nakretki = os.path.join(train_dir, 'Nakretki')
val_nakretki = os.path.join(val_dir, 'Nakretki')
test_nakretki = os.path.join(test_dir, 'Nakretki')

train_papieros = os.path.join(train_dir, 'Papieros')
val_papieros = os.path.join(val_dir, 'Papieros')
test_papieros = os.path.join(test_dir, 'Papieros')

train_piwo = os.path.join(train_dir, 'Piwo')
val_piwo = os.path.join(val_dir, 'Piwo')
test_piwo = os.path.join(test_dir, 'Piwo')

train_pudelko = os.path.join(train_dir, 'Pudelko')
val_pudelko = os.path.join(val_dir, 'Pudelko')
test_pudelko = os.path.join(test_dir, 'Pudelko')

train_puszka = os.path.join(train_dir, 'PuszkaMetal')
val_puszka = os.path.join(val_dir, 'PuszkaMetal')
test_puszka = os.path.join(test_dir, 'PuszkaMetal')

train_slomka = os.path.join(train_dir, 'SlomkaPlast')
val_slomka = os.path.join(val_dir, 'SlomkaPlast')
test_slomka = os.path.join(test_dir, 'SlomkaPlast')

train_widelec = os.path.join(train_dir, 'Widelec')
val_widelec = os.path.join(val_dir, 'Widelec')
test_widelec = os.path.join(test_dir, 'Widelec')

train_woreczek = os.path.join(train_dir, 'Woreczek')
val_woreczek = os.path.join(val_dir, 'Woreczek')
test_woreczek = os.path.join(test_dir, 'Woreczek')
'''
# Modul do kopiowania
'''
for j in range (20):
    fnames = ['Widelec{}.jpg' .format(i) for i in range(j*150, j*150+100)]
    for fname in fnames:
        image = os.path.join('Widelec', fname)
        dst = os.path.join(train_widelec, fname)
        shutil.copyfile(image, dst)

    fnames = ['Widelec{}.jpg' .format(i) for i in range(j*150+100, j*150+125)]
    for fname in fnames:
        image = os.path.join('Widelec', fname)
        dst = os.path.join(val_widelec, fname)
        shutil.copyfile(image, dst)

    fnames = ['Widelec{}.jpg' .format(i) for i in range(j*150+125, j*150+150)]
    for fname in fnames:
        image = os.path.join('Widelec', fname)
        dst = os.path.join(test_widelec, fname)
        shutil.copyfile(image, dst)
'''
#print('liczba obrazow treningowych: ', len(os.listdir(train_widelec)))
#print('liczba obrazow walidacyjnych: ', len(os.listdir(val_widelec)))
#print('liczba obrazow testowych: ', len(os.listdir(test_widelec)))

# Tworzenie generatorow, ktore beda wykorzystywaly uprzednio stworzone foldery z danymi
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(300, 300), batch_size=20, class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(val_dir, target_size=(300, 300), batch_size=20, class_mode='categorical')

'''
# Tworzenie modelu, ktory bedzie trenowany na treningowej bazie zdjec smieci
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(300,300,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

# Trenowanie modelu
history=model.fit(train_generator,steps_per_epoch=100,epochs=15,verbose=1,validation_data=validation_generator, validation_steps=20)
model.save('models/nz_3_notcomplete_but_improved_15_epoch_v2.h5')
'''
# Best model: 'Inz_3_notcomplete_but_improved_15_epoch_v2.h5'
model = models.load_model('models/Inz_3_notcomplete_but_improved_15_epoch_v2.h5')
model.summary()
from PIL import Image
import numpy as np
img = Image.open('examples/Kubeczek3294.jpg')
#img.show()
img = img.resize((300,300))
# img.show()
img = np.reshape(img,[1,300,300,3])
print(model.predict(img))

'''
# Tworzenie wykresow dokladnosci i straty
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Dokladnosc trenowania')
plt.plot(epochs, val_acc, 'b', label='Dokladnosc walidacji')
plt.title('Dokladnosc trenowania i walidacji')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Strata trenowania')
plt.plot(epochs, val_loss, 'b', label='Strata walidacji')
plt.title('Strata trenowania i walidacji')
plt.legend()

plt.show()
'''