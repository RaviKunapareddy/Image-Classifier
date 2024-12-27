import tensorflow as tf
import main


# Load the saved model
model_path = "cifar10_cnn_model.h5"
model = tf.keras.models.load_model(model_path)
#print("Model loaded successfully.")

class_pred = main.classify_image(main.test_images[589])
main.show_image_with_prediction(main.test_images[589], class_pred)

