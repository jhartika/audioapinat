import numpy as np

def main():
    
    
    train_data_file = "data/mel/train_data.npy"
    train_labels_file = "data/mel/train_labels.npy"
    test_data_file = "data/mel/test_data.npy"
    test_labels_file = "data/mel/test_labels.npy"
    
    train_data = np.load(train_data_file)
    train_labels = np.load(train_labels_file)
    test_data = np.load(test_data_file)
    test_labels = np.load(test_labels_file)

    from keras.layers import  Flatten, Dense, Dropout, GRU, Conv2D, MaxPool2D, Reshape, BatchNormalization
    from keras.optimizers import Adam, SGD
    from keras.models import Sequential
    from keras.utils import np_utils
    
    train_labels = np_utils.to_categorical(train_labels, 34)
    test_labels = np_utils.to_categorical(test_labels, 34)

#    train_data = train_data[..., np.newaxis]
#    test_data  = test_data[..., np.newaxis]
    
#    model = Sequential()
#    
#    model.add(GRU(64, return_sequences=True, activation='relu',input_shape=(40,44)))
#    model.add(BatchNormalization())
#    
#    model.add(GRU(64, return_sequences=True, activation='relu'))
#    model.add(BatchNormalization())
#    
#    model.add(Flatten())
#    model.add(Dense(34, activation='softmax'))
    
    model = Sequential()
    
    model.add(GRU(64, return_sequences=True, activation='relu',
                  dropout=0.5, recurrent_dropout=0.5,input_shape=(40,44)))
    model.add(BatchNormalization())
    
    model.add(GRU(64, return_sequences=True, activation='relu',
                  dropout=0.5, recurrent_dropout=0.5))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(34, activation='softmax'))

    model.summary()
    
    optim = Adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optim,
                  metrics=['accuracy'])
    
    model.fit(train_data, train_labels, epochs = 500, batch_size=5, validation_data = [test_data, test_labels])
    score = model.evaluate(test_data, test_labels, batch_size=128)
    print(score[1])
    
    
if __name__ == '__main__':
    main()
