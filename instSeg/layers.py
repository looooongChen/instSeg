from tensorflow import keras

def semantic_layer(classes)
    return keras.layers.Conv2D(filters=classes, activation='softmax')

def contour
            if m == 'contour':
                outlayer = keras.layers.Conv2D(filters=1, 
                                               kernel_size=3, padding='same', activation='sigmoid', 
                                               kernel_initializer='he_normal', 
                                               name='out_contour')
            if m == 'edt':
                activation = 'sigmoid' if self.config.edt_loss == 'binary_crossentropy' else 'linear'
                outlayer = keras.layers.Conv2D(filters=1, 
                                               kernel_size=3, padding='same', activation=activation, 
                                               kernel_initializer='he_normal',
                                               name='out_edt')
            if m == 'edt_flow':
                outlayer = keras.layers.Conv2D(filters=2, 
                                               kernel_size=3, padding='same', activation='linear', 
                                               kernel_initializer='he_normal',
                                               name='out_edt_flow')
            if m == 'embedding':
                outlayer = keras.layers.Conv2D(filters=self.config.embedding_dim, 
                                               kernel_size=3, padding='same', activation='linear', 
                                               kernel_initializer='he_normal', 
                                               name='out_embedding')