import tensorflow as tf
from tensorflow import keras

from models.metrics import loss_mse

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=10,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data
    
        #xu = (tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation="nearest")(x[0]),tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation="nearest")(x[1]))
        #xu = (tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x[0]),tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x[1]))
        # Forward pass of teacher
        teacher_logits = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_logits = self.student(x, training=True) 
            student_predictions = tf.nn.softmax(student_logits)
            #student_logits = self.student.get_layer('euclidean__distance').output
            
            # Compute losses
            classification_loss = self.student_loss_fn(y, student_predictions)

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(teacher_logits / self.temperature),
                    tf.nn.softmax(student_logits / self.temperature),
                )* self.temperature**2
            )

            loss = ((1-self.alpha) * classification_loss) + ((self.alpha) * distillation_loss)

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": classification_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        [xs,xq], y = data

        # Compute predictions
        student_logits = self.student([xs,xq], training=False)
        student_predictions = tf.nn.softmax(student_logits)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, student_predictions)

        # Update the metrics.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results