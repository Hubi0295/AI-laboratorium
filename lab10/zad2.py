from keras.api.models import Model
from keras.api.layers import Input, Dense, Reshape, BatchNormalization, Lambda, Flatten
from keras.api.optimizers import Adam
from keras.api.datasets import fashion_mnist
from matplotlib import pyplot as plt
def show_pictures(arrs):
    arr_cnt = arrs.shape[0]
    fig, axes = plt.subplots(1, arr_cnt, figsize=(5 * arr_cnt, arr_cnt))
    for axis, pic in zip(axes, arrs):
        axis.imshow(pic.squeeze(), cmap='gray')
        axis.axis('off')
    fig.tight_layout()
    return fig  

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Flatten the input to shape (784,)
X_train = X_train.reshape(X_train.shape[0], 28 * 28)
X_test = X_test.reshape(X_test.shape[0], 28 * 28)

act_func = 'selu'
hidden_dims = 64

encoder_layers = [
    BatchNormalization(),
    Dense(512, activation=act_func),
    Dense(128, activation=act_func),
    Dense(64, activation=act_func),
    Dense(hidden_dims, activation=act_func)
]

tensor = encoder_input = Input(shape=(28*28,))

for layer in encoder_layers:
    tensor = layer(tensor)
encoder_output = tensor
encoder = Model(inputs=encoder_input, outputs=encoder_output)

# Modify the decoder to flatten its output to match the target shape (784,)
decoder_layers = [
    Dense(128, activation=act_func),
    Dense(512, activation=act_func),
    Dense(784, activation='sigmoid'),  # output shape (784,)
    Lambda(lambda x: x * 255)           # scale to [0, 255]
]

decoder_input = tensor = Input(shape=(hidden_dims,))
for layer in decoder_layers:
    tensor = layer(tensor)
decoder_output = tensor
decoder = Model(inputs=decoder_input, outputs=decoder_output)

aec_output = decoder(encoder(encoder_input))
gen_autoencoder = Model(inputs=encoder_input, outputs=aec_output)

learning_rate = 0.00001
gen_autoencoder.compile(optimizer=Adam(learning_rate), loss='MeanSquaredError')

# Now the training data is properly reshaped
gen_autoencoder.fit(x=X_train, y=X_train, validation_data=(X_test, X_test), batch_size=256, epochs=5)


def show_pictures(arrs):
    arr_cnt = arrs.shape[0]
    fig, axes = plt.subplots(1, arr_cnt, figsize=(5 * arr_cnt, arr_cnt))
    if arr_cnt == 1:  # If there's only one image, axes won't be an array.
        axes = [axes]
    
    for axis, pic in zip(axes, arrs):
        # Reshape each image to (28, 28) before showing
        axis.imshow(pic.reshape(28, 28), cmap='gray')  # Reshape to (28, 28)
        axis.axis('off')
    
    fig.tight_layout()
    return fig


from keras import backend as K
from keras.api.layers import Lambda
import tensorflow as tf

def adding_noise(tensor):
    # Use TensorFlow's tf.random.normal to generate noise
    noise = tf.random.normal(shape=tf.shape(tensor), mean=0.0, stddev=1.5)
    return tensor + noise

# Specify output_shape for the Lambda layer
noised_encoder_output = Lambda(adding_noise, output_shape=(64,))(encoder_output)

augmenter_output = decoder(noised_encoder_output)
augmenter = Model(inputs=encoder_input, outputs=augmenter_output)

def filter_data(data, iteration_num):
    augmented_data = data.copy()
    for i in range(iteration_num):
        augmented_data = gen_autoencoder.predict(augmented_data)  
    return augmented_data


start = 50
end = start + 10

for i in range(10):
    test_for_augm = X_train[i*10:i*10+10,...]
    augmented_data = test_for_augm.copy()
    show_pictures(test_for_augm)
    augmented_data = augmenter.predict(augmented_data)
    show_pictures(augmented_data)
    augmented_data = filter_data(augmented_data, 5)
    show_pictures(augmented_data)


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
original_image = X_train[0]
original_image = np.expand_dims(original_image,axis=-1)
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)
image_generator = datagen.flow(
    np.expand_dims(original_image, axis=0),
    batch_size=1
)
num_images = 10
generated_images = np.zeros((num_images,28,28))
for i in range(num_images):
    augmented_image = next(image_generator)[0]
    generated_images[i] = augmented_image[:,:,]
