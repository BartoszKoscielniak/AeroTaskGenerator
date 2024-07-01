import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import expit

class AeroTask:
    def __init__(self, task_id, duration):
        self.task_id = task_id
        self.duration = int(np.floor(duration))

    def __repr__(self):
        return f"Task({self.task_id}, Duration: {self.duration} min)"

class AeroTaskGenerator:
    def __init__(self, num_tasks, mean, std_dev, function_type='normal', **kwargs):
        self.num_tasks = num_tasks
        
        if function_type == 'normal':
            self.normal_mean = mean
            self.normal_std_dev = std_dev
        elif function_type == 'gaussian':
            self.gaussian_mean = mean
            self.gaussian_std_dev = std_dev
            self.gaussian_sigma = kwargs.get('sigma', 1.0)
        elif function_type == 'sigmoid':
            self.sigmoid_center = mean
            self.sigmoid_scale = std_dev
        else:
            raise ValueError(f"Unsupported function type: {function_type}")
        
        self.function_type = function_type
        self.tasks = []
        self.task_limits = []
        self.task_matrix = None
        self.max_duration = 30

    def generate_histogram_function(self):
        x = np.arange(1, self.num_tasks + 1)
        
        if self.function_type == 'normal':
            y = norm.pdf(x, loc=self.normal_mean, scale=self.normal_std_dev)
        elif self.function_type == 'gaussian':
            y = np.exp(-((x - self.gaussian_mean) ** 2 / (2 * self.gaussian_sigma ** 2)))
        elif self.function_type == 'sigmoid':
            y = expit((x - self.sigmoid_center) / self.sigmoid_scale)
        else:
            raise ValueError(f"Unsupported function type: {self.function_type}")

        y = y / max(y) * self.max_duration
        self.task_limits = np.floor(y).astype(int) 

    def generate_tasks(self):
        self.tasks = []
        self.task_matrix = np.zeros((self.num_tasks, self.max_duration))
        
        for task_id in range(1, self.num_tasks + 1):
            max_duration_minutes = self.task_limits[task_id - 1]
            self.task_matrix[task_id - 1, :max_duration_minutes] = 1
        
        for task_id in range(1, self.num_tasks + 1):
            duration = np.sum(self.task_matrix[task_id - 1, :])
            task = AeroTask(task_id, duration)
            self.tasks.append(task)

    def save_tasks_to_csv(self, filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Task ID", "Duration (min)"])
            for task in self.tasks:
                writer.writerow([task.task_id, task.duration])

    def load_tasks_from_csv(self, filename):
        self.tasks = []
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                task_id, duration = int(row[0]), int(np.floor(float(row[1])))
                self.tasks.append(AeroTask(task_id, duration))

    def plot_histogram_function(self):
        x = np.arange(1, self.num_tasks + 1)
        y = self.task_limits
        plt.plot(x, y, marker='o', linestyle='-', color='b')
        plt.title("Generated Histogram Function")
        plt.xlabel("Task Number")
        plt.ylabel("Max Duration [min]")
        plt.xticks(np.arange(1, self.num_tasks + 1, step=1))
        plt.show()

    def plot_tasks_with_histogram(self):
        x = np.arange(1, self.num_tasks + 1)
        y = self.task_limits

        plt.plot(x, y, marker='o', linestyle='-', color='b', label='Histogram Function')

        task_durations = [task.duration for task in self.tasks]
        plt.bar(range(1, self.num_tasks + 1), task_durations, color='r', alpha=0.5, label='Tasks')

        plt.title("Tasks with Histogram Function")
        plt.xlabel("Task Number")
        plt.ylabel("Duration [min]")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    num_tasks = int(input("Enter the number of tasks(e.g. 15): "))
    function_type = input("Enter the function type (e.g. normal, gaussian, sigmoid): ")
    
    if function_type == 'normal':
        mean = float(input("Enter the mean (e.g. 7): "))
        std_dev = float(input("Enter the standard deviation(e.g. 4): "))
        generator = AeroTaskGenerator(num_tasks, mean, std_dev, function_type=function_type)
    elif function_type == 'gaussian':
        mean = float(input("Enter the mean(e.g. 7): "))
        std_dev = float(input("Enter the standard deviation(e.g. 3): "))
        sigma = float(input("Enter the sigma(e.g. 2.0): "))
        generator = AeroTaskGenerator(num_tasks, mean, std_dev, function_type=function_type, sigma=sigma)
    elif function_type == 'sigmoid':
        center = float(input("Enter the center(e.g. 5): "))
        scale = float(input("Enter the scale(e.g. 1): "))
        generator = AeroTaskGenerator(num_tasks, center, scale, function_type=function_type)
    else:
        raise ValueError("Unsupported function type.")

    generator.generate_histogram_function()
    generator.generate_tasks()
    filename = f"tasks_{function_type}.csv"
    generator.save_tasks_to_csv(filename)
    generator.load_tasks_from_csv(filename)
    generator.plot_histogram_function()
    generator.plot_tasks_with_histogram()

    for task in generator.tasks:
        print(task)
