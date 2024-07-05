import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.simpledialog import askfloat

# Model zadania
class Task:
    def __init__(self, id, duration):
        self.id = id
        self.duration = duration

class AeroTaskGenerator:
    def __init__(self, num_tasks=1000, max_task_time=30):
        self.num_tasks = num_tasks
        self.max_task_time = max_task_time
        # Tworzymy ndarray z wartosciami [1, 2, ..., 30]
        self.task_times = np.arange(1, max_task_time + 1)

    # Funkcja sigmoid
    def sigmoid(self, x, a=1, b=0):
        return 1 / (1 + np.exp(-a * (x - b)))

    # Funkcja sigmoid odwrócona
    def inverted_sigmoid(self, x, a=1, b=0):
        return ((1 / (1 + np.exp(-a * (x - b)))) - 1) * (-1)

    # Funkcja Gaussa
    def gauss(self, x, mu=15, sigma=5):
        # np.exp oblicza funkcje wykladnicza elementow tablicy
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Funkcja wykładnicza
    def exponential(self, x, lambda_=1):
        return lambda_ * np.exp(-lambda_ * x)

    # Funkcja log-normalna
    def lognormal(self, x, mu=0, sigma=1):
        return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2)

    # Funkcja Butterwortha
    def butterworth(self, x, cutoff=0.5, order=2):
        return 1 / (1 + (x / cutoff)**(2*order))

    # Funkcja środkowo przepustowa
    def bandpass(self, x, a1=1, b1=0, a2=1, b2=0):
        sigmoid_1 = self.sigmoid(x, a1, b1)
        inverted_sigmoid_2 = self.inverted_sigmoid(x, a2, b2)
        return np.maximum(sigmoid_1, inverted_sigmoid_2)

    # Generowanie zadan na podstawie wybranego rozkladu
    def generate_tasks(self, func='sigmoid', **params):
        if func == 'sigmoid':
            y = self.sigmoid(self.task_times, **params)
        elif func == 'gauss':
            y = self.gauss(self.task_times, **params)
        elif func == 'exponential':
            y = self.exponential(self.task_times, **params)
        elif func == 'lognormal':
            y = self.lognormal(self.task_times, **params)
        elif func == 'butterworth':
            y = self.butterworth(self.task_times, **params)
        elif func == 'bandpass':
            y = self.bandpass(self.task_times, **params)
        else:
            raise ValueError("Unsupported function. Use 'sigmoid', 'gauss', 'exponential', 'lognormal', 'butterworth', or 'bandpass'.")

        # Normalizuje tablice wartosci aby uzyskac rozklad prawdopodobienstwa wystapienia zadania z okreslonym czasem trwania
        hist_func = y / np.sum(y)

        # Generowanie czasów trwania zadan na podstawie rozkładu
        # self.task_times tablica mozliwych czasow trwania zadan
        # self.num_tasks liczba zadan do wygenerowania
        # hist_func okresla rozklad prawdopodobienstwa wyboru czasu trwania zadania
        task_durations = np.random.choice(self.task_times, self.num_tasks, p=hist_func)
        tasks = [Task(i + 1, duration) for i, duration in enumerate(task_durations)]
        return tasks, hist_func

    # Wizualizacja rozkladu zadan
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

    # Zapisanie zadan do pliku CSV
    def save_tasks_to_csv(self, tasks, filename):
        data = {'Task ID': [task.id for task in tasks], 'Duration': [task.duration for task in tasks]}
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    # Odczytanie zadan z pliku CSV
    def read_tasks_from_csv(self, filename):
        df = pd.read_csv(filename)
        tasks = [Task(row['Task ID'], row['Duration']) for _, row in df.iterrows()]
        return tasks

    def get_task_summary(self, tasks):
        durations = [task.duration for task in tasks]
        unique, counts = np.unique(durations, return_counts=True)
        task_summary = pd.DataFrame({'Duration': unique, 'Number of Tasks': counts})
        task_summary['Value %'] = round(task_summary['Number of Tasks'] / self.num_tasks * 100, 2)
        return task_summary

    def save_summary_to_csv(self, summary, filename):
        summary.to_csv(filename, index=False)

    def read_summary_from_csv(self, filename):
        summary = pd.read_csv(filename)
        return summary

# Klasa GUI
class TaskGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AeroTask Generator")

        self.frame = ttk.Frame(root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.create_widgets()
        self.generator = AeroTaskGenerator()  # Inicjalizacja generatora z domyslnymi wartosciami

    # Tworzenie interfejsu uzytkownika
    def create_widgets(self):
        ttk.Label(self.frame, text="Number of Tasks:").grid(column=0, row=0, sticky=tk.W)
        self.num_tasks = tk.StringVar()
        ttk.Entry(self.frame, textvariable=self.num_tasks).grid(column=1, row=0, sticky=(tk.W, tk.E))

        ttk.Label(self.frame, text="Function Type:").grid(column=0, row=1, sticky=tk.W)
        self.function_type = tk.StringVar()
        ttk.Combobox(self.frame, textvariable=self.function_type, values=["sigmoid", "gaussian", "exponential", "lognormal", "butterworth", "bandpass"]).grid(column=1, row=1, sticky=(tk.W, tk.E))

        self.parameter_frame = ttk.Frame(self.frame, padding="10")
        self.parameter_frame.grid(column=0, row=2, columnspan=2, sticky=(tk.W, tk.E))

        ttk.Button(self.frame, text="Generate Tasks", command=self.generate_tasks).grid(column=0, row=3, columnspan=2, sticky=(tk.W, tk.E))
        ttk.Button(self.frame, text="Load Tasks from CSV", command=self.load_tasks_from_csv).grid(column=0, row=4, columnspan=2, sticky=(tk.W, tk.E))
        ttk.Button(self.frame, text="Save Tasks to CSV", command=self.save_tasks_to_csv).grid(column=0, row=5, columnspan=2, sticky=(tk.W, tk.E))
        ttk.Button(self.frame, text="Plot Distribution", command=self.plot_distribution).grid(column=0, row=6, columnspan=2, sticky=(tk.W, tk.E))
        ttk.Button(self.frame, text="Show Task Summary", command=self.show_task_summary).grid(column=0, row=7, columnspan=2, sticky=(tk.W, tk.E))
        ttk.Button(self.frame, text="Save Summary to CSV", command=self.save_summary_to_csv).grid(column=0, row=8, columnspan=2, sticky=(tk.W, tk.E))
        ttk.Button(self.frame, text="Load Summary from CSV", command=self.load_summary_from_csv).grid(column=0, row=9, columnspan=2, sticky=(tk.W, tk.E))
        ttk.Button(self.frame, text="Exit", command=self.root.quit).grid(column=0, row=10, columnspan=2, sticky=(tk.W, tk.E))

        self.task_list = tk.Text(self.frame, height=15, width=50)
        self.task_list.grid(column=0, row=11, columnspan=2, sticky=(tk.W, tk.E))

        task_scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.task_list.yview)
        self.task_list['yscrollcommand'] = task_scrollbar.set
        task_scrollbar.grid(column=2, row=11, sticky=(tk.N, tk.S))

        self.summary_list = tk.Text(self.frame, height=15, width=50)
        self.summary_list.grid(column=0, row=12, columnspan=2, sticky=(tk.W, tk.E))

        summary_scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.summary_list.yview)
        self.summary_list['yscrollcommand'] = summary_scrollbar.set
        summary_scrollbar.grid(column=2, row=12, sticky=(tk.N, tk.S))

    # Generowanie zadan na podstawie wprowadzonych parametrow
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

        elif function_type == 'exponential':
            lambda_ = askfloat("Input", "Enter the lambda (e.g., 1):")
            self.tasks, self.hist_func = self.generator.generate_tasks(func='exponential', lambda_=lambda_)

        elif function_type == 'lognormal':
            mu = askfloat("Input", "Enter the mu (e.g., 0):")
            sigma = askfloat("Input", "Enter the sigma (e.g., 1):")
            self.tasks, self.hist_func = self.generator.generate_tasks(func='lognormal', mu=mu, sigma=sigma)

        elif function_type == 'butterworth':
            cutoff = askfloat("Input", "Enter the cutoff frequency (e.g., 0.5):")
            order = askfloat("Input", "Enter the filter order (e.g., 2):")
            self.tasks, self.hist_func = self.generator.generate_tasks(func='butterworth', cutoff=cutoff, order=order)

        elif function_type == 'bandpass':
            a1 = askfloat("Input", "Enter the a1 (e.g., 1):")
            b1 = askfloat("Input", "Enter the b1 (e.g., 15):")
            a2 = askfloat("Input", "Enter the a2 (e.g., 1):")
            b2 = askfloat("Input", "Enter the b2 (e.g., 15):")
            self.tasks, self.hist_func = self.generator.generate_tasks(func='bandpass', a1=a1, b1=b1, a2=a2, b2=b2)

        else:
            messagebox.showerror("Error", "Unsupported function type.")

        self.display_tasks()

    # Wyswietlanie zadan w GUI aplikacji
    def display_tasks(self):
        self.task_list.delete(1.0, tk.END)
        for task in self.tasks:
            self.task_list.insert(tk.END, f"Task {task.id}: {task.duration} min\n")

    # Zapisywanie zadan do pliku CSV
    def save_tasks_to_csv(self):
        if not hasattr(self, 'tasks') or not self.tasks:
            messagebox.showerror("Error", "No tasks to save.")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if filename:
            self.generator.save_tasks_to_csv(self.tasks, filename)

    # Wczytywanie zadan z pliku CSV
    def load_tasks_from_csv(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            self.tasks = self.generator.read_tasks_from_csv(filename)
            self.display_tasks()

    # Wizualizacja rozkladu zadan
    def plot_distribution(self):
        if not hasattr(self, 'tasks') or not self.tasks:
            messagebox.showerror("Error", "No tasks to plot.")
            return
        self.generator.plot_distribution(self.tasks, self.hist_func, self.function_type.get())

    def show_task_summary(self):
        if not hasattr(self, 'tasks') or not self.tasks:
            messagebox.showerror("Error", "No tasks to summarize.")
            return
        summary = self.generator.get_task_summary(self.tasks)
        self.summary_list.delete(1.0, tk.END)
        self.summary_list.insert(tk.END, summary.to_string(index=False))

    def save_summary_to_csv(self):
        if not hasattr(self, 'tasks') or not self.tasks:
            messagebox.showerror("Error", "No tasks to summarize.")
            return
        summary = self.generator.get_task_summary(self.tasks)
        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if filename:
            self.generator.save_summary_to_csv(summary, filename)

    def load_summary_from_csv(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            summary = self.generator.read_summary_from_csv(filename)
            self.summary_list.delete(1.0, tk.END)
            self.summary_list.insert(tk.END, summary.to_string(index=False))

# Uruchomienie aplikacji
if __name__ == "__main__":
    root = tk.Tk()
    app = TaskGeneratorApp(root)
    root.mainloop()
