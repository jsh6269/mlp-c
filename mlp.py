import numpy as np
import time
from PIL import Image, ImageDraw

class MLP:
    def __init__(self):
        self.hidden_size1 = 64
        self.hidden_size2 = 64

        np.random.seed(5489)

        scale_input = np.sqrt(2.0 / 2)
        scale_h = np.sqrt(2.0 / self.hidden_size1)
        scale_out = np.sqrt(2.0 / self.hidden_size2)

        self.input_weights = np.random.normal(0, scale_input, (self.hidden_size1, 2))
        self.hidden_weights = np.random.normal(0, scale_h, (self.hidden_size2, self.hidden_size1))
        self.output_weights = np.random.normal(0, scale_out, self.hidden_size2)

        self.input_biases = np.zeros(self.hidden_size1)
        self.hidden_biases = np.zeros(self.hidden_size2)
        self.output_bias = np.float64(0)

        self.training_time = 0
        self.accuracy = 0

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1.0 - sig)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1.0 - np.square(np.tanh(x))

    def forward(self, x, y):
        input_vec = np.array([x, y])

        hidden1_raw = self.input_weights @ input_vec + self.input_biases
        hidden1 = self.tanh(hidden1_raw)

        hidden2_raw = self.hidden_weights @ hidden1 + self.hidden_biases
        hidden2 = self.tanh(hidden2_raw)

        output_raw = self.output_weights @ hidden2 + self.output_bias
        output = self.sigmoid(output_raw)

        return output

    def train(self, data, labels, epochs=2000, learning_rate=0.08):
        start = time.time()
        data = np.array(data)
        labels = np.array(labels)

        for epoch in range(epochs):
            total_loss = 0.0
            for idx in range(len(data)):
                x, y = data[idx]
                target = labels[idx]
                input_vec = np.array([x, y])

                hidden1_raw = self.input_weights @ input_vec + self.input_biases
                hidden1 = self.tanh(hidden1_raw)

                hidden2_raw = self.hidden_weights @ hidden1 + self.hidden_biases
                hidden2 = self.tanh(hidden2_raw)

                output_raw = self.output_weights @ hidden2 + self.output_bias
                output = self.sigmoid(output_raw)

                loss = 0.5 * (target - output) ** 2
                total_loss += loss

                d_output = (output - target) * output * (1 - output)
                
                d_hidden2 = d_output * self.output_weights * self.tanh_derivative(hidden2_raw)
                d_hidden1 = self.tanh_derivative(hidden1_raw) * (self.hidden_weights.T @ d_hidden2)

                self.output_weights -= learning_rate * d_output * hidden2
                self.output_bias -= learning_rate * d_output

                self.hidden_weights -= learning_rate * np.outer(d_hidden2, hidden1)
                self.hidden_biases -= learning_rate * d_hidden2

                self.input_weights -= learning_rate * np.outer(d_hidden1, input_vec)
                self.input_biases -= learning_rate * d_hidden1

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / len(data):.6f}")

        self.training_time = int((time.time() - start) * 1000)
        self.accuracy = self.get_accuracy(data, labels)

    def get_accuracy(self, data, labels):
        correct = 0
        for i in range(len(data)):
            pred = self.forward(data[i][0], data[i][1])
            if (pred >= 0.5 and labels[i] == 1) or (pred < 0.5 and labels[i] == 0):
                correct += 1
        return correct / len(data)

    def get_training_time(self):
        return self.training_time

def save_visualization(mlp, data, labels, filename):
    width = 300
    height = 250
    step = 5
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    for x in range(0, width, step):
        for y in range(0, height, step):
            input_x = x / width
            input_y = y / height
            output = mlp.forward(input_x, input_y)

            r = int((1 - output) * 135 + output * 255)
            g = int((1 - output) * 206 + output * 105)
            b = int((1 - output) * 235 + output * 180)

            for dx in range(step):
                for dy in range(step):
                    if x + dx < width and y + dy < height:
                        draw.point((x + dx, y + dy), fill=(r, g, b))

    radius = 6
    for i in range(len(data)):
        x = int(data[i][0] * width)
        y = int(data[i][1] * height)
        color = (255, 0, 0) if labels[i] == 1 else (0, 0, 255)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)

    image.save(filename)

def main():
    data = [
        [11, 21], [28, 17], [19, 15], [6, 9], [7, 1], [8, 10], 
        [14, 21], [15, 14], [14, 17], [15, 20], [28, 21], [9, 13], 
        [19, 23], [29, 11], [23, 18], [2, 19], [27, 15], [6, 6], 
        [12, 18], [0, 4], [13, 14], [3, 20], [6, 2], [3, 24], 
        [9, 10], [14, 10], [8, 14], [10, 6], [13, 15], [9, 16], 
        [12, 7], [20, 22], [17, 16], [19, 16], [11, 18], [19, 11], 
        [0, 10], [13, 16], [16, 24], [23, 15], [12, 18], [0, 0], 
        [22, 12], [8, 13], [25, 9], [26, 21], [6, 16], [28, 16], 
        [15, 6], [4, 10], [26, 5], [6, 8], [15, 14], [18, 7], 
        [3, 24], [1, 15], [9, 16], [4, 23], [2, 20], [15, 12], 
        [29, 11], [13, 19], [23, 22], [23, 1], [14, 17], [9, 24], 
        [16, 12], [21, 17], [9, 7], [21, 16], [21, 12], [12, 24], 
        [18, 13], [16, 8], [6, 1], [9, 7], [18, 5], [8, 14], 
        [12, 21], [8, 22]
    ]
    labels = [
        1, 1, 0, 0, 0, 0, 
        1, 0, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 0, 
        1, 0, 1, 1, 0, 1, 
        0, 0, 1, 0, 1, 1, 
        0, 1, 0, 0, 1, 0, 
        0, 1, 1, 1, 1, 0, 
        0, 1, 0, 1, 1, 1, 
        0, 0, 0, 0, 0, 0, 
        1, 0, 1, 1, 1, 0, 
        1, 1, 1, 0, 1, 1, 
        0, 1, 0, 1, 0, 1, 
        0, 0, 0, 0, 0, 1, 
        1, 1
    ]

    data = np.array(data, dtype=np.float64)
    data[:, 0] /= 30.0
    data[:, 1] /= 25.0

    mlp = MLP()
    mlp.train(data, labels, epochs=2000)
    save_visualization(mlp, data, labels, "visualized_python.png")

    print("--------------------------------")
    print("Saved Result: visualized_python.png")
    print(f"Train Accuracy: {int(mlp.get_accuracy(data, labels) * 100)}%")
    print(f"Train Time: {mlp.get_training_time()}ms")

if __name__ == "__main__":
    main()
