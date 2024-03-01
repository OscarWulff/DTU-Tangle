import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

root = tk.Tk()
root.title("Circular Graph")

fig = Figure(figsize=(4, 4), dpi=100)
ax = fig.add_subplot(111)
ax.add_patch(plt.Circle((0.5, 0.5), 0.4, color='blue', fill=False))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()

root.mainloop()
