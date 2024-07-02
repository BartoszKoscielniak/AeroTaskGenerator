import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Task:
    def __init__(self, id, duration):
        self.id = id
        self.duration = duration

class AeroTaskGenerator:
    def __init__(self, num_tasks=1000, max_task_time=30):
        self.num_tasks = num_tasks
        self.max_task_time = max_task_time
        self.task_times = np.arange(1, max_task_time + 1)

    def sigmoid(self, x, a=1, b=0):
        return 1 / (1 + np.exp(-a * (x - b)))

    def gauss(self, x, mu=15, sigma=5):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def generate_tasks(self, func='sigmoid', **params):
        if func == 'sigmoid':
            y = self.sigmoid(self.task_times, **params)
        elif func == 'gauss':
            y = self.gauss(self.task_times, **params)
        else:
            raise ValueError("Unsupported function. Use 'sigmoid', 'gauss', or 'normal'.")

        hist_func = y / np.sum(y)

        # Generate task durations based on the given probability distribution (hist_func)
        # - `self.task_times` is the array of possible task durations.
        # - `self.num_tasks` is the number of tasks to generate.
        # - `p=hist_func` specifies the probability distribution for choosing each duration.
        task_durations = np.random.choice(self.task_times, self.num_tasks, p=hist_func)
        
        tasks = [Task(i+1, duration) for i, duration in enumerate(task_durations)]
        return tasks, hist_func

    def plot_distribution(self, task_durations, hist_func, func_name):
        durations = [task.duration for task in task_durations]
        hist, bin_edges = np.histogram(durations, bins=self.max_task_time, range=(1, self.max_task_time + 1))

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(self.task_times, hist_func, label=f'{func_name.capitalize()} Histogram Function')
        plt.title(f'{func_name.capitalize()} Histogram Function')
        plt.xlabel('Task Duration (minutes)')
        plt.ylabel('Probability')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.bar(bin_edges[:-1], hist, alpha=0.6, color='g', label='Generated Task Distribution')
        plt.plot(bin_edges[:-1], hist, 'o-', label='Generated Task Histogram')
        plt.title('Generated Task Distribution')
        plt.xlabel('Task Duration (minutes)')
        plt.ylabel('Number of Tasks')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def print_task_lengths(self, task_durations):
        for task in task_durations:
            print(f"Task {task.id}: {task.duration} min")

    def save_tasks_to_csv(self, tasks, filename):
        data = {'Task ID': [task.id for task in tasks], 'Duration': [task.duration for task in tasks]}
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    def read_tasks_from_csv(self, filename):
        df = pd.read_csv(filename)
        tasks = [Task(row['Task ID'], row['Duration']) for _, row in df.iterrows()]
        return tasks

if __name__ == "__main__":

    num_tasks = int(input("Enter the number of tasks(e.g. 500): "))
    function_type = input("Enter the function type (e.g. gaussian, sigmoid): ")

    if function_type == 'gaussian':
        mu = float(input("Enter the mu(e.g. 15): "))
        sigma = float(input("Enter the sigma(e.g. 5): "))

        generator = AeroTaskGenerator(num_tasks=num_tasks, max_task_time=30)
        tasks_gauss, hist_func_gauss = generator.generate_tasks(func='gauss', mu=mu, sigma=sigma)
        generator.plot_distribution(tasks_gauss, hist_func_gauss, func_name='gauss')
        generator.print_task_lengths(tasks_gauss)
        generator.save_tasks_to_csv(tasks_gauss, 'tasks_gauss.csv')

    elif function_type == 'sigmoid':
        a = float(input("Enter the a(e.g. 0.5): "))
        b = float(input("Enter the b(e.g. 15): "))

        generator = AeroTaskGenerator(num_tasks=num_tasks, max_task_time=30)
        tasks_sigmoid, hist_func_sigmoid = generator.generate_tasks(func='sigmoid', a=a, b=b)
        generator.plot_distribution(tasks_sigmoid, hist_func_sigmoid, func_name='sigmoid')
        generator.print_task_lengths(tasks_sigmoid)
        generator.save_tasks_to_csv(tasks_sigmoid, 'tasks_sigmoid.csv')

    else:
        raise ValueError("Unsupported function type.")