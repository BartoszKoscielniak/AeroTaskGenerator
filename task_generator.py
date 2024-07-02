import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.simpledialog import askfloat

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
        task_durations = np.random.choice(self.task_times, self.num_tasks, p=hist_func)
        tasks = [Task(i + 1, duration) for i, duration in enumerate(task_durations)]
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

    def save_tasks_to_csv(self, tasks, filename):
        data = {'Task ID': [task.id for task in tasks], 'Duration': [task.duration for task in tasks]}
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    def read_tasks_from_csv(self, filename):
        df = pd.read_csv(filename)
        tasks = [Task(row['Task ID'], row['Duration']) for _, row in df.iterrows()]
        return tasks

class TaskGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AeroTask Generator")

        self.frame = ttk.Frame(root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.create_widgets()
        self.generator = AeroTaskGenerator()  # Initialize the generator with default values

    def create_widgets(self):
        ttk.Label(self.frame, text="Number of Tasks:").grid(column=0, row=0, sticky=tk.W)
        self.num_tasks = tk.StringVar()
        ttk.Entry(self.frame, textvariable=self.num_tasks).grid(column=1, row=0, sticky=(tk.W, tk.E))

        ttk.Label(self.frame, text="Function Type:").grid(column=0, row=1, sticky=tk.W)
        self.function_type = tk.StringVar()
        ttk.Combobox(self.frame, textvariable=self.function_type, values=["sigmoid", "gaussian"]).grid(column=1, row=1, sticky=(tk.W, tk.E))

        self.parameter_frame = ttk.Frame(self.frame, padding="10")
        self.parameter_frame.grid(column=0, row=2, columnspan=2, sticky=(tk.W, tk.E))

        ttk.Button(self.frame, text="Generate Tasks", command=self.generate_tasks).grid(column=0, row=3, columnspan=2, sticky=(tk.W, tk.E))
        ttk.Button(self.frame, text="Load Tasks from CSV", command=self.load_tasks_from_csv).grid(column=0, row=4, columnspan=2, sticky=(tk.W, tk.E))
        ttk.Button(self.frame, text="Save Tasks to CSV", command=self.save_tasks_to_csv).grid(column=0, row=5, columnspan=2, sticky=(tk.W, tk.E))
        ttk.Button(self.frame, text="Plot Distribution", command=self.plot_distribution).grid(column=0, row=6, columnspan=2, sticky=(tk.W, tk.E))
        ttk.Button(self.frame, text="Exit", command=self.root.quit).grid(column=0, row=7, columnspan=2, sticky=(tk.W, tk.E))

        self.task_list = tk.Text(self.frame, height=15, width=50)
        self.task_list.grid(column=0, row=8, columnspan=2, sticky=(tk.W, tk.E))

    def generate_tasks(self):
        num_tasks = int(self.num_tasks.get())
        function_type = self.function_type.get()

        self.generator = AeroTaskGenerator(num_tasks=num_tasks, max_task_time=30)

        if function_type == 'gaussian':
            mu = askfloat("Input", "Enter the mu (e.g., 15):")
            sigma = askfloat("Input", "Enter the sigma (e.g., 5):")
            self.tasks, self.hist_func = self.generator.generate_tasks(func='gauss', mu=mu, sigma=sigma)

        elif function_type == 'sigmoid':
            a = askfloat("Input", "Enter the a (e.g., 0.5):")
            b = askfloat("Input", "Enter the b (e.g., 15):")
            self.tasks, self.hist_func = self.generator.generate_tasks(func='sigmoid', a=a, b=b)

        else:
            messagebox.showerror("Error", "Unsupported function type.")

        self.display_tasks()

    def display_tasks(self):
        self.task_list.delete(1.0, tk.END)
        for task in self.tasks:
            self.task_list.insert(tk.END, f"Task {task.id}: {task.duration} min\n")

    def save_tasks_to_csv(self):
        if not hasattr(self, 'tasks') or not self.tasks:
            messagebox.showerror("Error", "No tasks to save.")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if filename:
            self.generator.save_tasks_to_csv(self.tasks, filename)

    def load_tasks_from_csv(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            self.tasks = self.generator.read_tasks_from_csv(filename)
            self.display_tasks()

    def plot_distribution(self):
        if not hasattr(self, 'tasks') or not self.tasks:
            messagebox.showerror("Error", "No tasks to plot.")
            return
        self.generator.plot_distribution(self.tasks, self.hist_func, self.function_type.get())

if __name__ == "__main__":
    root = tk.Tk()
    app = TaskGeneratorApp(root)
    root.mainloop()
