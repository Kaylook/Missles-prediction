import cv2
import numpy as np
import mss
import time
import tkinter as tk
from collections import deque

# === Глобальные переменные ===
target_history = deque(maxlen=10)
time_stamps = deque(maxlen=10)
overlay_window = None
explosion_window = None

# GUI элементы (будут инициализированы позже)
status_window = None
time_label = None
dist_label = None
capture_label = None

# Параметры
TOTAL_TIME = 0.0
START_TIME = 0.0
RUNNING = False
LAST_AIM_POINT = (800, 400)
MISSILE_SPEED_MPS = 0.0

# === Поиск треугольника ===
def find_triangle(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv_gray = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(inv_gray, 128, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        if len(approx) == 3 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if 0.5 < aspect_ratio < 2.0:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
    return None

# === Обновление оверлея ===
def update_overlay(x, y, filled=False):
    global overlay_window
    size = 40
    offset = size // 2

    if overlay_window is None:
        overlay_window = tk.Tk()
        overlay_window.overrideredirect(True)
        overlay_window.wm_attributes("-topmost", True)
        overlay_window.wm_attributes("-transparentcolor", "white")
        overlay_window.geometry(f"{size}x{size}+{x-offset}+{y-offset}")

        canvas = tk.Canvas(overlay_window, width=size, height=size, bg='white', highlightthickness=0)
        canvas.pack()
        if filled:
            canvas.create_oval(0, 0, size, size, fill="red", outline="")
        else:
            canvas.create_oval(5, 5, size-5, size-5, outline="red", width=3, fill="")
        overlay_window.update_idletasks()
    else:
        overlay_window.geometry(f"+{x-offset}+{y-offset}")
        canvas = overlay_window.winfo_children()[0]
        canvas.delete("all")
        if filled:
            canvas.create_oval(0, 0, size, size, fill="red", outline="")
        else:
            canvas.create_oval(5, 5, size-5, size-5, outline="red", width=3, fill="")
        overlay_window.update_idletasks()

# === Основная логика обновления (вызывается через after) ===
def update_guidance():
    global RUNNING, LAST_AIM_POINT

    if not RUNNING:
        return

    current_time = time.time()
    elapsed = current_time - START_TIME
    remaining = max(0.0, TOTAL_TIME - elapsed)

    # Обновляем GUI (в основном потоке — безопасно)
    dist_label.config(text=f"Расстояние: {remaining / TOTAL_TIME * 1000:.0f} м" if TOTAL_TIME > 0 else "Расстояние: -- м")
    time_label.config(text=f"До взрыва: {remaining:.1f} с")

    if remaining <= 0:
        # 💥 ВЗРЫВ
        update_overlay(LAST_AIM_POINT[0], LAST_AIM_POINT[1], filled=True)
        capture_label.config(text="Цель: --", fg="gray")
        RUNNING = False
        return

    # Поиск цели
    try:
        sct = mss.mss()
        monitor = sct.monitors[1]
        screenshot = np.array(sct.grab(monitor))
        frame = screenshot[:, :, :3]
        target_pos = find_triangle(frame)
    except Exception:
        target_pos = None

    if target_pos is not None:
        target_history.append(target_pos)
        time_stamps.append(current_time)
        capture_label.config(text="Цель: захвачена", fg="green")
    else:
        capture_label.config(text="Цель: не найдена", fg="red")

    # Обновляем упреждение раз в секунду
    if hasattr(update_guidance, 'last_update'):
        if current_time - update_guidance.last_update >= 1.0:
            aim_point = target_pos
            if target_pos is not None and len(target_history) >= 2:
                dt_total = time_stamps[-1] - time_stamps[0]
                if dt_total > 0:
                    dx = target_history[-1][0] - target_history[0][0]
                    dy = target_history[-1][1] - target_history[0][1]
                    vx, vy = dx / dt_total, dy / dt_total
                    pred_time = min(5.0, remaining)
                    aim_x = int(target_pos[0] + vx * pred_time)
                    aim_y = int(target_pos[1] + vy * pred_time)
                    aim_point = (aim_x, aim_y)

            if aim_point:
                LAST_AIM_POINT = aim_point
                update_overlay(aim_point[0], aim_point[1], filled=False)

            update_guidance.last_update = current_time
    else:
        update_guidance.last_update = current_time

    # Запланировать следующее обновление
    status_window.after(100, update_guidance)  # каждые 100 мс

# === Стартовое окно ===
def show_start_form():
    root = tk.Tk()
    root.title("Запуск ракеты")
    root.geometry("300x180")
    root.resizable(False, False)

    tk.Label(root, text="Расстояние до цели (м):").pack(pady=5)
    dist_entry = tk.Entry(root, justify='center')
    dist_entry.insert(0, "1000")
    dist_entry.pack()

    tk.Label(root, text="Скорость припаса (км/ч):").pack(pady=5)
    speed_entry = tk.Entry(root, justify='center')
    speed_entry.insert(0, "1200")
    speed_entry.pack()

    def on_launch():
        try:
            distance_m = float(dist_entry.get())
            speed_kmh = float(speed_entry.get())
            if distance_m <= 0 or speed_kmh <= 0:
                raise ValueError
        except:
            tk.messagebox.showerror("Ошибка", "Введите корректные числа!")
            return

        global TOTAL_TIME, START_TIME, RUNNING, MISSILE_SPEED_MPS
        MISSILE_SPEED_MPS = speed_kmh * 1000 / 3600
        TOTAL_TIME = distance_m / MISSILE_SPEED_MPS
        START_TIME = time.time()
        RUNNING = True

        root.destroy()
        create_status_window()

    tk.Button(root, text="🚀 ПУСК", command=on_launch, bg="lightgreen", font=("Arial", 12)).pack(pady=15)
    root.mainloop()

# === Создание окна статуса ===
def create_status_window():
    global status_window, time_label, dist_label, capture_label
    status_window = tk.Tk()
    status_window.title("Статус наведения")
    status_window.geometry("280x140")
    status_window.resizable(False, False)
    status_window.wm_attributes("-topmost", True)

    time_label = tk.Label(status_window, text="До взрыва: -- с", font=("Courier", 11))
    dist_label = tk.Label(status_window, text="Расстояние: -- м", font=("Courier", 11))
    capture_label = tk.Label(status_window, text="Цель: --", font=("Arial", 12), fg="gray")

    time_label.pack(pady=3)
    dist_label.pack(pady=3)
    capture_label.pack(pady=3)

    tk.Label(status_window, text="Контур — упреждение\nЗакрашенный — взрыв", font=("Arial", 8), fg="gray").pack(pady=5)

    # Запуск обновления в основном потоке
    status_window.after(100, update_guidance)
    status_window.mainloop()

# === Запуск ===
if __name__ == "__main__":
    show_start_form()