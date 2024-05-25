import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.stats import norm

class AeroTask:
    def __init__(self, task_id, duration):
        self.task_id = task_id
        self.duration = duration

    def __repr__(self):
        return f"Task({self.task_id}, Duration: {self.duration} h)"

class AeroTaskGenerator:
    def __init__(self, num_tasks, mean, std_dev):
        self.num_tasks = num_tasks
        self.mean = mean
        self.std_dev = std_dev
        self.tasks = []
        self.task_limits = []

    def generate_histogram_function(self):
        x = np.arange(1, self.num_tasks + 1)
        y = norm.pdf(x, loc=self.mean, scale=self.std_dev)
        y = y / max(y) * 8  # Normalize to maximum 8 hours
        self.task_limits = np.floor(y).astype(int)  # Round down to the nearest hour

    def generate_tasks(self):
        self.tasks = []
        for task_id in range(1, self.num_tasks + 1):  # Task IDs start from 1
            max_duration_hours = self.task_limits[task_id - 1]
            duration = max_duration_hours  # Use the entire available duration
            task = AeroTask(task_id, duration)
            self.tasks.append(task)

    def save_tasks_to_csv(self, filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Task ID", "Duration (h)"])
            for task in self.tasks:
                writer.writerow([task.task_id, task.duration])

    def load_tasks_from_csv(self, filename):
        self.tasks = []
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                task_id, duration = int(row[0]), int(row[1])
                self.tasks.append(AeroTask(task_id, duration))

    def plot_histogram_function(self):
        x = np.arange(1, self.num_tasks + 1)
        y = self.task_limits
        plt.plot(x, y, marker='o', linestyle='-', color='b')
        plt.title("Generated Histogram Function")
        plt.xlabel("Task Number")
        plt.ylabel("Max Duration [h]")
        plt.xticks(np.arange(1, self.num_tasks + 1, step=1))
        plt.show()

    def plot_tasks_with_histogram(self):
        x = np.arange(1, self.num_tasks + 1)
        y = self.task_limits

        # Plot histogram function
        plt.plot(x, y, marker='o', linestyle='-', color='b', label='Histogram Function')

        # Plot tasks as bar chart
        task_durations = [task.duration for task in self.tasks]  # Duration already in hours
        plt.bar(range(1, self.num_tasks + 1), task_durations, color='r', alpha=0.5, label='Tasks')

        plt.title("Tasks with Histogram Function")
        plt.xlabel("Task Number")
        plt.ylabel("Duration [h]")
        plt.legend()
        plt.show()

# Użycie aplikacji AeroTaskGenerator
if __name__ == "__main__":
    num_tasks = 10
    mean_task_number = 5  # Mean task number (arbitrary for example)
    std_dev_task_number = 2  # Standard deviation in task numbers

    generator = AeroTaskGenerator(num_tasks, mean_task_number, std_dev_task_number)
    generator.generate_histogram_function()
    generator.generate_tasks()
    generator.save_tasks_to_csv("tasks.csv")
    generator.load_tasks_from_csv("tasks.csv")

    generator.plot_histogram_function()
    generator.plot_tasks_with_histogram()

    for task in generator.tasks:
        print(task)
