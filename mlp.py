import numpy as np
import time
from PIL import Image, ImageDraw

class MLP:
    def __init__(self):
        self.hidden_size = 4
        
        # C++의 mt19937과 동일한 시드값 사용
        np.random.seed(5489)  # mt19937의 기본 시드값
        
        # float64(C++의 double과 동일) 타입 명시
        self.input_weights = np.array(np.random.normal(0, 1, (8,)) * 0.1, dtype=np.float64)
        self.hidden_weights = np.array(np.random.normal(0, 1, (4,)) * 0.1, dtype=np.float64)
        self.input_bias = np.float64(np.random.normal(0, 1) * 0.1)
        self.hidden_bias = np.float64(np.random.normal(0, 1) * 0.1)
        
        self.training_time = 0
        self.accuracy = 0

    def sigmoid(self, x):
        # 명시적으로 float64 사용
        return np.float64(1.0) / (np.float64(1.0) + np.exp(-x, dtype=np.float64))

    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (np.float64(1.0) - sig)

    def forward(self, x, y):
        # 입력을 float64로 변환
        x = np.float64(x)
        y = np.float64(y)
        
        # 입력층 -> 은닉층
        hidden = np.zeros(self.hidden_size, dtype=np.float64)
        for i in range(self.hidden_size):
            hidden[i] = self.sigmoid(x * self.input_weights[i] + 
                                   y * self.input_weights[i + self.hidden_size] + 
                                   self.input_bias)
        
        # 은닉층 -> 출력층
        output = np.float64(0)
        for i in range(self.hidden_size):
            output += hidden[i] * self.hidden_weights[i]
        output = self.sigmoid(output + self.hidden_bias)
        
        return output

    def train(self, data, labels, epochs=1000, learning_rate=0.1):
        # learning_rate를 float64로 변환
        learning_rate = np.float64(learning_rate)
        start_time = time.time()
        
        for epoch in range(epochs):
            total_loss = np.float64(0)
            
            for i in range(len(data)):
                x = np.float64(data[i][0])
                y = np.float64(data[i][1])
                target = np.float64(labels[i])
                
                # Forward pass
                hidden = np.zeros(self.hidden_size, dtype=np.float64)
                hidden_raw = np.zeros(self.hidden_size, dtype=np.float64)
                
                for j in range(self.hidden_size):
                    hidden_raw[j] = (x * self.input_weights[j] + 
                                   y * self.input_weights[j + self.hidden_size] + 
                                   self.input_bias)
                    hidden[j] = self.sigmoid(hidden_raw[j])
                
                output_raw = np.float64(0)
                for j in range(self.hidden_size):
                    output_raw += hidden[j] * self.hidden_weights[j]
                output_raw += self.hidden_bias
                output = self.sigmoid(output_raw)
                
                # 손실 계산
                loss = np.float64(0.5) * (target - output) * (target - output)
                total_loss += loss
                
                # Backward pass
                d_output = (output - target) * self.sigmoid_derivative(output_raw)
                
                # 은닉층 -> 출력층 가중치 업데이트
                for j in range(self.hidden_size):
                    self.hidden_weights[j] -= learning_rate * d_output * hidden[j]
                self.hidden_bias -= learning_rate * d_output
                
                # 입력층 -> 은닉층 가중치 업데이트
                for j in range(self.hidden_size):
                    d_hidden = (d_output * self.hidden_weights[j] * 
                              self.sigmoid_derivative(hidden_raw[j]))
                    self.input_weights[j] -= learning_rate * d_hidden * x
                    self.input_weights[j + self.hidden_size] -= learning_rate * d_hidden * y
                self.input_bias -= learning_rate * d_output
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(data):.8f}")
        
        self.training_time = int((time.time() - start_time) * 1000)  # ms로 변환
        self.accuracy = self.get_accuracy(data, labels)

    def get_accuracy(self, data, labels):
        correct = 0
        for i in range(len(data)):
            output = self.forward(data[i][0], data[i][1])
            predicted = 1 if output >= 0.5 else 0
            if predicted == labels[i]:
                correct += 1
        return correct / len(data)

    def get_training_time(self):
        return self.training_time

def save_visualization(mlp, data, labels, filename):
    width = 300
    height = 250
    step = 5
    
    # 이미지 생성
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # 결정 경계 시각화 (배경 먼저)
    for x in range(0, width, step):
        for y in range(0, height, step):
            input_x = x / width
            input_y = y / height
            output = mlp.forward(input_x, input_y)
            
            # C++와 동일한 색상 계산
            r = int((1 - output) * 135 + output * 255)
            g = int((1 - output) * 206 + output * 105)
            b = int((1 - output) * 235 + output * 180)
            
            # 픽셀 채우기
            for dx in range(step):
                for dy in range(step):
                    if x + dx < width and y + dy < height:
                        draw.point((x + dx, y + dy), fill=(r, g, b))
    
    # 학습 데이터 포인트 그리기
    radius = 6
    for i in range(len(data)):
        x = int(data[i][0] * width)
        y = int(data[i][1] * height)
        color = (255, 0, 0) if labels[i] == 1 else (0, 0, 255)
        
        # 원 그리기
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
    
    # 이미지 저장
    image.save(filename)

def main():
    start_time = time.time()
    
    print(f"Size of np.float64: {np.dtype(np.float64).itemsize} bytes")
    
    test_value = np.float64(0.1)
    bytes_array = test_value.tobytes()
    print("Internal representation of 0.1:", " ".join(f"{b:02x}" for b in bytes_array))
    print("--------------------------------")

    # 학습 데이터 생성 (C++와 동일)
    data = [
        [10, 23], [4, 21], [8, 17], [17, 22], [3, 12], [9, 12], [16, 14], [21, 20], [26, 22],  # upper
        [4, 4], [13, 7], [16, 6], [14, 3], [22, 8], [24, 5], [28, 6], [27, 14]                # lower
    ]
    
    labels = [1] * 9 + [0] * 8  # upper: 1, lower: 0
    
    # 데이터 정규화
    data = np.array(data, dtype=float)
    data[:, 0] /= 30.0  # x축: 0~30 기준
    data[:, 1] /= 25.0  # y축: 0~25 기준
    
    # MLP 모델 생성 및 학습
    mlp = MLP()
    mlp.train(data, labels)
    
    # 결과 시각화
    save_visualization(mlp, data, labels, "visualized_python.png")
    
    # 결과 출력
    total_time = int((time.time() - start_time) * 1000)
    train_time = mlp.get_training_time()
    print("--------------------------------")
    print("Saved Result: visualized_python.png")
    print(f"Train Accuracy: {int(100 * mlp.get_accuracy(data, labels))}%")
    print(f"Train Time: {train_time}ms")
    print(f"Overhead: {total_time - train_time}ms")
    print(f"Total Time: {total_time}ms")
    print()

if __name__ == "__main__":
    main()
