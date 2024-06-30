import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import expit  

class AeroTask:
    def __init__(self, task_id, duration):
        self.task_id = task_id
        self.duration = duration

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

        y = y / max(y) * 30
        self.task_limits = np.floor(y).astype(int) 

    def generate_tasks(self):
        self.tasks = []
        for task_id in range(1, self.num_tasks + 1):
            max_duration_minutes = self.task_limits[task_id - 1]
            duration = max_duration_minutes  
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
                task_id, duration = int(row[0]), int(row[1])
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
    num_tasks = 15
    
    #Normal distribution
    normal_mean = 7
    normal_std_dev = 4
    generator_normal = AeroTaskGenerator(num_tasks, normal_mean, normal_std_dev, function_type='normal')
    generator_normal.generate_histogram_function()
    generator_normal.generate_tasks()
    generator_normal.save_tasks_to_csv("tasks_normal.csv")
    generator_normal.load_tasks_from_csv("tasks_normal.csv")
    generator_normal.plot_histogram_function()
    generator_normal.plot_tasks_with_histogram()

    for task in generator_normal.tasks:
        print(task)

    #Gaussian function
    gaussian_mean = 7
    gaussian_std_dev = 3
    gaussian_sigma = 2.0
    generator_gaussian = AeroTaskGenerator(num_tasks, gaussian_mean, gaussian_std_dev, function_type='gaussian', sigma=gaussian_sigma)
    generator_gaussian.generate_histogram_function()
    generator_gaussian.generate_tasks()
    generator_gaussian.save_tasks_to_csv("tasks_gaussian.csv")
    generator_gaussian.load_tasks_from_csv("tasks_gaussian.csv")
    generator_gaussian.plot_histogram_function()
    generator_gaussian.plot_tasks_with_histogram()

    for task in generator_gaussian.tasks:
        print(task)

    #Sigmoid function
    sigmoid_center = 5
    sigmoid_scale = 1
    generator_sigmoid = AeroTaskGenerator(num_tasks, sigmoid_center, sigmoid_scale, function_type='sigmoid')
    generator_sigmoid.generate_histogram_function()
    generator_sigmoid.generate_tasks()
    generator_sigmoid.save_tasks_to_csv("tasks_sigmoid.csv")
    generator_sigmoid.load_tasks_from_csv("tasks_sigmoid.csv")
    generator_sigmoid.plot_histogram_function()
    generator_sigmoid.plot_tasks_with_histogram()

    for task in generator_sigmoid.tasks:
        print(task)
