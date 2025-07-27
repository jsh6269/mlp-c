import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from PIL import Image, ImageDraw

torch.manual_seed(5489)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden_size1 = 64
        self.hidden_size2 = 64

        self.fc1 = nn.Linear(2, self.hidden_size1)
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.fc3 = nn.Linear(self.hidden_size2, 1)

        # He Initialization (수동)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='tanh')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='tanh')
        nn.init.zeros_(self.fc2.bias)
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='sigmoid')
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def train_model(model, data, labels, epochs=2000, lr=0.32):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    start = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(data).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    training_time = int((time.time() - start) * 1000)
    return training_time


def get_accuracy(model, data, labels):
    with torch.no_grad():
        model.eval()
        preds = model(data).squeeze()
        binary_preds = (preds >= 0.5).float()
        accuracy = (binary_preds == labels).float().mean().item()
        return accuracy


def save_visualization(model, data, labels, filename):
    width = 300
    height = 250
    step = 5
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    with torch.no_grad():
        for x in range(0, width, step):
            for y in range(0, height, step):
                input_x = x / width
                input_y = y / height
                input_tensor = torch.tensor([[input_x, input_y]], dtype=torch.float32)
                output = model(input_tensor).item()

                r = int((1 - output) * 135 + output * 255)
                g = int((1 - output) * 206 + output * 105)
                b = int((1 - output) * 235 + output * 180)

                for dx in range(step):
                    for dy in range(step):
                        if x + dx < width and y + dy < height:
                            draw.point((x + dx, y + dy), fill=(r, g, b))

    radius = 6
    for i in range(len(data)):
        x = int(data[i][0].item() * width)
        y = int(data[i][1].item() * height)
        color = (255, 0, 0) if labels[i].item() == 1 else (0, 0, 255)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)

    image.save(filename)


def main():
    data = np.array([
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
    ], dtype=np.float32)

    labels = np.array([
        1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0,
        0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
        1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1,
        0, 0, 0, 0, 0, 1, 1, 1
    ], dtype=np.float32)

    # Normalize
    data[:, 0] /= 30.0
    data[:, 1] /= 25.0

    # Tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.float32)

    model = MLP()

    # 배치학습에 대한 보정으로 학습률을 0.32로 높게 설정
    train_time = train_model(model, data_tensor, label_tensor, epochs=2000, lr=0.32)
    accuracy = get_accuracy(model, data_tensor, label_tensor)

    save_visualization(model, data_tensor, label_tensor, "visualized_torch.png")

    print("--------------------------------")
    print("Saved Result: visualized_torch.png")
    print(f"Train Accuracy: {int(accuracy * 100)}%")
    print(f"Train Time: {train_time}ms")

if __name__ == "__main__":
    main()
