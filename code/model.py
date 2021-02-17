import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
from tensorflow.keras import layers, models, losses

from utils import *

def get_session(gpu_fraction=0.5):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
 
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

class Model:

    def __init__(self):
        self.nn = None
        self.star_w = []
        self.star_b = []
        self.fisher_w = []
        self.fisher_b = []

    def create_model(self, classes=10):
        self.nn = models.Sequential([
            layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", input_shape=(28, 28, 1)),
            layers.LeakyReLU(alpha=0.01),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            layers.Dropout(0.05),
            layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same"),
            layers.LeakyReLU(alpha=0.01),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            layers.Dropout(0.05),
            layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same"),
            layers.LeakyReLU(alpha=0.01),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            layers.Dropout(0.05),
            layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same"),
            layers.LeakyReLU(alpha=0.01),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            layers.Dropout(0.05),
            layers.Flatten(), 
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.05),                                                   
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.05),                                                   
            layers.Dense(10, activation='softmax')
        ])

        self.nn.summary()
        self.nn.compile(optimizer='Adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

    def EWC_loss(self):
        
        def loss(y_true, y_pred) :
            lamda = tf.constant(1.0, dtype=tf.float32)
            cce = losses.CategoricalCrossentropy()
            L = cce(y_true, y_pred)

            trained_params = self.nn.trainable_weights
            weight = [trained_params[i] for i in range(0, len(trained_params), 2)]
            bias = [trained_params[i] for i in range(1, len(trained_params), 2)]

            for f, s, w in zip(self.fisher_w, self.star_w, weight) :
                L = L + (lamda/2) * tf.reduce_sum(tf.multiply(f, tf.square(w - s)))

            for f, s, b in zip(self.fisher_b, self.star_b, bias) :
                L = L + (lamda/2) * tf.reduce_sum(tf.multiply(f, tf.square(b - s)))

            return L
        
        return loss

    def transfer_model(self, model, prev_tasks):
        self.nn = models.clone_model(model.nn)
        self.nn.set_weights(model.nn.get_weights())

        self.compute_fisher(prev_tasks)
        self.compute_star()

        # Change loss function
        self.nn.compile(optimizer='Adam',
                        loss=self.EWC_loss(),
                        metrics=['accuracy'])

    def compute_star(self):
        # Used for saving optimal weights after most recent task training
        self.star_w = []
        self.star_b = []

        trained_params = self.nn.trainable_weights
        weight = [trained_params[i] for i in range(0, len(trained_params), 2)]
        bias = [trained_params[i] for i in range(1, len(trained_params), 2)]

        for w, b in zip(weight, bias):
            self.star_w.append(w)
            self.star_b.append(b)

    def compute_fisher(self, tasks):
        self.fisher_w = []
        self.fisher_b = []

        output = self.nn.output
        trained_params = self.nn.trainable_weights
        gradients = K.gradients(output, trained_params)

        x = []
        for task in tasks :
            n = task['x_train'].shape[0]
            samples = np.random.choice(range(n), 500, replace=False)
            [x.append(d) for d in task['x_train'][samples]]
            
        x = np.array(x, dtype=np.float32)
        m = x.shape[0]

        sess = tf.InteractiveSession()
        sess.run(tf.compat.v1.global_variables_initializer())

        evaluated_gradients = np.square(sess.run(gradients, feed_dict={self.nn.input:x}))

        for i in range(0, len(trained_params), 2):
            self.fisher_w.append(evaluated_gradients[i]/m)
            self.fisher_b.append(evaluated_gradients[i+1]/m)

        sess.close()

    def save(self, ckpt):
        if self.nn == None :
            print("Do training first!")
            return 0

        self.nn.save("../param/"+ckpt)

    def predict(self, x, ckpt):
        if self.nn == None :
            if not os.path.isfile("../param/"+ckpt) :
                print("Do training first!")
                return 0

            self.create_model(classes=10)
            self.nn.load_weights("../param/"+ckpt)

        y_pred = self.nn.predict_classes(x, batch_size=200)
        return y_pred

    def train(self, tasks):

        if self.nn == None :
            self.create_model(classes=10)
        
        out_epoch = EpochLogger(tasks=tasks)
        
        self.nn.fit(x = tasks[-1]['x_train'],
                    y = tasks[-1]['y_train'],
                    epochs = 10,
                    verbose = 0,
                    batch_size=500,
                    callbacks=[out_epoch])

        return out_epoch.get_result()

if __name__ == '__main__' :
    get_session(gpu_fraction=0.3)
    np.random.seed(9876)




