import tensorflow as tf
'''
-- Long Chen, LfB, RWTH Aachen University --
tensroflow 2.x model: UNet, UNet_d7
U-Net: Convolutional Networks for Biomedical Image Segmentation
'''

class RecurrentUNnet(tf.keras.Model):

    def __init__(self,
                 output_channel=1,
                 filters=32,
                 dropout_rate=0.2,
                 batch_normalization=False,
                 stateful=True,
                 name='RecurrentUNnet',
                 **kwargs):

        super(RecurrentUNnet, self).__init__(name=name, **kwargs)
        self.batch_normalization=batch_normalization
        self.stateful = stateful

        # encode_conv1
        self.Conv1_1 = tf.keras.layers.ConvLSTM2D(filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Conv1_2 = tf.keras.layers.ConvLSTM2D(filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Pool1 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))
        # encode_conv2
        if batch_normalization:
            self.BatchNorm2 = tf.keras.layers.BatchNormalization()
        self.Drop2 = tf.keras.layers.Dropout(dropout_rate)
        self.Conv2_1 = tf.keras.layers.ConvLSTM2D(2*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Conv2_2 = tf.keras.layers.ConvLSTM2D(2*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Pool2 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))
        # encode_conv3
        if batch_normalization:
            self.BatchNorm3 = tf.keras.layers.BatchNormalization()
        self.Drop3 = tf.keras.layers.Dropout(dropout_rate)
        self.Conv3_1 = tf.keras.layers.ConvLSTM2D(4*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Conv3_2 = tf.keras.layers.ConvLSTM2D(4*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Pool3 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))
        # encode_conv4
        if batch_normalization:
            self.BatchNorm4 = tf.keras.layers.BatchNormalization()
        self.Drop4 = tf.keras.layers.Dropout(dropout_rate)
        self.Conv4_1 = tf.keras.layers.ConvLSTM2D(8*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Conv4_2 = tf.keras.layers.ConvLSTM2D(8*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Pool4 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))
        # encode_conv5
        if batch_normalization:
            self.BatchNorm5 = tf.keras.layers.BatchNormalization()
        self.Drop5 = tf.keras.layers.Dropout(dropout_rate)
        self.Conv5_1 = tf.keras.layers.ConvLSTM2D(16*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Conv5_2 = tf.keras.layers.ConvLSTM2D(16*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        # decode_conv6
        self.Up6 = tf.keras.layers.UpSampling3D(size=(1, 2, 2))
        self.Merge6 = tf.keras.layers.Concatenate(axis=-1)
        if batch_normalization:
            self.BatchNorm6 = tf.keras.layers.BatchNormalization()
        self.Drop6 = tf.keras.layers.Dropout(dropout_rate)
        self.Conv6_1 = tf.keras.layers.ConvLSTM2D(8*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Conv6_2 = tf.keras.layers.ConvLSTM2D(8*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        # decode_conv7
        self.Up7 = tf.keras.layers.UpSampling3D(size=(1, 2, 2))
        self.Merge7 = tf.keras.layers.Concatenate(axis=-1)
        if batch_normalization:
            self.BatchNorm7 = tf.keras.layers.BatchNormalization()
        self.Drop7 = tf.keras.layers.Dropout(dropout_rate)
        self.Conv7_1 = tf.keras.layers.ConvLSTM2D(4*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Conv7_2 = tf.keras.layers.ConvLSTM2D(4*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        # decode_conv8
        self.Up8 = tf.keras.layers.UpSampling3D(size=(1, 2, 2))
        self.Merge8 = tf.keras.layers.Concatenate(axis=-1)
        self.Drop8 = tf.keras.layers.Dropout(dropout_rate)
        if batch_normalization:
            self.BatchNorm8 = tf.keras.layers.BatchNormalization()
        self.Conv8_1 = tf.keras.layers.ConvLSTM2D(2*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Conv8_2 = tf.keras.layers.ConvLSTM2D(2*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        # decode_conv9
        self.Up9 = tf.keras.layers.UpSampling3D(size=(1, 2, 2))
        self.Merge9 = tf.keras.layers.Concatenate(axis=-1)
        self.Drop9 = tf.keras.layers.Dropout(dropout_rate)
        if batch_normalization:
            self.BatchNorm9 = tf.keras.layers.BatchNormalization()
        self.Conv9_1 = tf.keras.layers.ConvLSTM2D(filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Conv9_2 = tf.keras.layers.ConvLSTM2D(filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)

        self.Conv_out = tf.keras.layers.ConvLSTM2D(output_channel, 1, stateful=stateful, activation='relu', padding='same', return_sequences=False)

    def call(self, inputs):

        # conv1 
        conv1_1 = self.Conv1_1(inputs)
        conv1_2 = self.Conv1_2(conv1_1)
        pool1 = self.Pool1(conv1_2)
        # conv2
        drop2 = self.Drop2(self.BatchNorm2(pool1)) if self.batch_normalization else self.Drop2(pool1)
        conv2_1 = self.Conv2_1(drop2)
        conv2_2 = self.Conv2_2(conv2_1)
        pool2 = self.Pool2(conv2_2)
        # conv3
        drop3 = self.Drop3(self.BatchNorm3(pool2)) if self.batch_normalization else self.Drop3(pool2)
        conv3_1 = self.Conv3_1(drop3)
        conv3_2 = self.Conv3_2(conv3_1)
        pool3 = self.Pool3(conv3_2)
        # conv4
        drop4 = self.Drop4(self.BatchNorm4(pool3)) if self.batch_normalization else self.Drop4(pool3)
        conv4_1 = self.Conv4_1(drop4)
        conv4_2 = self.Conv4_2(conv4_1)
        pool4 = self.Pool4(conv4_2)
        # conv5
        drop5 = self.Drop5(self.BatchNorm5(pool4)) if self.batch_normalization else self.Drop5(pool4)
        conv5_1 = self.Conv5_1(drop5)
        conv5_2 = self.Conv5_2(conv5_1)
        # conv6
        merge6 = self.Merge6([conv4_2, self.Up6(conv5_2)])
        drop6 = self.Drop6(self.BatchNorm6(merge6)) if self.batch_normalization else self.Drop6(merge6)
        conv6_1 = self.Conv6_1(drop6)
        conv6_2 = self.Conv6_2(conv6_1)
        # conv7
        merge7 = self.Merge7([conv3_2, self.Up7(conv6_2)])
        drop7 = self.Drop7(self.BatchNorm7(merge7)) if self.batch_normalization else self.Drop7(merge7)
        conv7_1 = self.Conv7_1(drop7)
        conv7_2 = self.Conv7_2(conv7_1)
        # conv8
        merge8 = self.Merge8([conv2_2, self.Up8(conv7_2)])
        drop8 = self.Drop8(self.BatchNorm8(merge8)) if self.batch_normalization else self.Drop8(merge8)
        conv8_1 = self.Conv8_1(drop8)
        conv8_2 = self.Conv8_2(conv8_1)
        # conv9
        merge9 = self.Merge9([conv1_2, self.Up9(conv8_2)])
        drop9 = self.Drop9(self.BatchNorm9(merge9)) if self.batch_normalization else self.Drop9(merge9)
        conv9_1 = self.Conv9_1(drop9)
        conv9_2 = self.Conv9_2(conv9_1)

        output = self.Conv_out(conv9_2)

        return output

    def reset_states(states=None):
        if self.stateful:
            self.Conv1_1.reset_states(states)
            self.Conv1_2.reset_states(states)
            self.Conv2_1.reset_states(states)
            self.Conv2_2.reset_states(states)
            self.Conv3_1.reset_states(states)
            self.Conv3_2.reset_states(states)
            self.Conv4_1.reset_states(states)
            self.Conv4_2.reset_states(states)
            self.Conv5_1.reset_states(states)
            self.Conv5_2.reset_states(states)
            self.Conv6_1.reset_states(states)
            self.Conv6_2.reset_states(states)
            self.Conv7_1.reset_states(states)
            self.Conv7_2.reset_states(states)
            self.Conv8_1.reset_states(states)
            self.Conv8_2.reset_states(states)
            self.Conv9_1.reset_states(states)
            self.Conv9_2.reset_states(states)
            self.Conv_out.reset_states(states)

class RecurrentUNnet_d7(tf.keras.Model):

    def __init__(self,
                 output_channel=1,
                 filters=32,
                 dropout_rate=0.2,
                 batch_normalization=False,
                 stateful=True,
                 name='UNRecurrentUNnet_d7',
                 **kwargs):

        super(RecurrentUNnet_d7, self).__init__(name=name, **kwargs)
        self.batch_normalization=batch_normalization
        self.stateful = stateful

        # encode_conv1
        self.Conv1_1 = tf.keras.layers.ConvLSTM2D(filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Conv1_2 = tf.keras.layers.ConvLSTM2D(filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Pool1 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))
        # encode_conv2
        if batch_normalization:
            self.BatchNorm2 = tf.keras.layers.BatchNormalization()
        self.Drop2 = tf.keras.layers.Dropout(dropout_rate)
        self.Conv2_1 = tf.keras.layers.ConvLSTM2D(2*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Conv2_2 = tf.keras.layers.ConvLSTM2D(2*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Pool2 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))
        # encode_conv3
        if batch_normalization:
            self.BatchNorm3 = tf.keras.layers.BatchNormalization()
        self.Drop3 = tf.keras.layers.Dropout(dropout_rate)
        self.Conv3_1 = tf.keras.layers.ConvLSTM2D(4*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Conv3_2 = tf.keras.layers.ConvLSTM2D(4*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Pool3 = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))
        # encode_conv4
        if batch_normalization:
            self.BatchNorm4 = tf.keras.layers.BatchNormalization()
        self.Drop4 = tf.keras.layers.Dropout(dropout_rate)
        self.Conv4_1 = tf.keras.layers.ConvLSTM2D(8*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Conv4_2 = tf.keras.layers.ConvLSTM2D(8*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        # decode_conv5
        self.Up5 = tf.keras.layers.UpSampling3D(size=(1, 2, 2))
        self.Merge5 = tf.keras.layers.Concatenate(axis=-1)
        if batch_normalization:
            self.BatchNorm5 = tf.keras.layers.BatchNormalization()
        self.Drop5 = tf.keras.layers.Dropout(dropout_rate)
        self.Conv5_1 = tf.keras.layers.ConvLSTM2D(4*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Conv5_2 = tf.keras.layers.ConvLSTM2D(4*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        # decode_conv6
        self.Up6 = tf.keras.layers.UpSampling3D(size=(1, 2, 2))
        self.Merge6 = tf.keras.layers.Concatenate(axis=-1)
        if batch_normalization:
            self.BatchNorm6 = tf.keras.layers.BatchNormalization()
        self.Drop6 = tf.keras.layers.Dropout(dropout_rate)
        self.Conv6_1 = tf.keras.layers.ConvLSTM2D(2*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Conv6_2 = tf.keras.layers.ConvLSTM2D(2*filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        # decode_conv7
        self.Up7 = tf.keras.layers.UpSampling3D(size=(1, 2, 2))
        self.Merge7 = tf.keras.layers.Concatenate(axis=-1)
        if batch_normalization:
            self.BatchNorm7 = tf.keras.layers.BatchNormalization()
        self.Drop7 = tf.keras.layers.Dropout(dropout_rate)
        self.Conv7_1 = tf.keras.layers.ConvLSTM2D(filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        self.Conv7_2 = tf.keras.layers.ConvLSTM2D(filters, 3, stateful=stateful, activation='relu', padding='same', return_sequences=True)
        
        self.Conv_out = tf.keras.layers.ConvLSTM2D(output_channel, 1, stateful=stateful, activation='relu', padding='same', return_sequences=False)

    def call(self, inputs):
        
        # conv1 
        conv1_1 = self.Conv1_1(inputs)
        conv1_2 = self.Conv1_2(conv1_1)
        pool1 = self.Pool1(conv1_2)
        # conv2
        drop2 = self.Drop2(self.BatchNorm2(pool1)) if self.batch_normalization else self.Drop2(pool1)
        conv2_1 = self.Conv2_1(drop2)
        conv2_2 = self.Conv2_2(conv2_1)
        pool2 = self.Pool2(conv2_2)
        # conv3
        drop3 = self.Drop3(self.BatchNorm3(pool2)) if self.batch_normalization else self.Drop3(pool2)
        conv3_1 = self.Conv3_1(drop3)
        conv3_2 = self.Conv3_2(conv3_1)
        pool3 = self.Pool3(conv3_2)
        # conv4
        drop4 = self.Drop4(self.BatchNorm4(pool3)) if self.batch_normalization else self.Drop4(pool3)
        conv4_1 = self.Conv4_1(drop4)
        conv4_2 = self.Conv4_2(conv4_1)
        # conv5
        merge5 = self.Merge5([conv3_2, self.Up6(conv4_2)])
        drop5 = self.Drop5(self.BatchNorm5(merge5)) if self.batch_normalization else self.Drop5(merge5)
        conv5_1 = self.Conv5_1(drop5)
        conv5_2 = self.Conv5_2(conv5_1)
        # conv6
        merge6 = self.Merge6([conv2_2, self.Up6(conv5_2)])
        drop6 = self.Drop6(self.BatchNorm6(merge6)) if self.batch_normalization else self.Drop6(merge6)
        conv6_1 = self.Conv6_1(drop6)
        conv6_2 = self.Conv6_2(conv6_1)
        # conv7
        merge7 = self.Merge7([conv1_2, self.Up7(conv6_2)])
        drop7 = self.Drop7(self.BatchNorm7(merge7)) if self.batch_normalization else self.Drop7(merge7)
        conv7_1 = self.Conv7_1(drop7)
        conv7_2 = self.Conv7_2(conv7_1)

        output = self.Conv_out(conv7_2)

    def reset_states(states=None):
        if self.stateful:
            self.Conv1_1.reset_states(states)
            self.Conv1_2.reset_states(states)
            self.Conv2_1.reset_states(states)
            self.Conv2_2.reset_states(states)
            self.Conv3_1.reset_states(states)
            self.Conv3_2.reset_states(states)
            self.Conv4_1.reset_states(states)
            self.Conv4_2.reset_states(states)
            self.Conv5_1.reset_states(states)
            self.Conv5_2.reset_states(states)
            self.Conv6_1.reset_states(states)
            self.Conv6_2.reset_states(states)
            self.Conv7_1.reset_states(states)
            self.Conv7_2.reset_states(states)
            self.Conv_out.reset_states(states)

        return output

if __name__ == "__main__":
    import numpy as np
    import os

    model = RecurrentUNnet_d7(output_channel=1, filters=32, dropout_rate=0.2, batch_normalization=True, name='UNet')

    @tf.function
    def trace_func():
        inputs = np.zeros((2,10,512,512,1)).astype(np.float32)
        outputs = model(inputs)
        print(outputs.shape) 
        return outputs

    # Set up logging.
    logdir = '.\\logs_check'
    writer = tf.summary.create_file_writer(logdir)

    # Bracket the function call with
    # tf.summary.trace_on() and tf.summary.trace_export().
    tf.summary.trace_on(graph=True, profiler=True)
    # Call only one tf.function when tracing.
    z = trace_func()
    with writer.as_default():
        tf.summary.trace_export(name="network_check", step=0, profiler_outdir=logdir)
