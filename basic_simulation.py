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

# Параметры полёта
TOTAL_TIME = 0.0
START_TIME = 0.0
RUNNING = False
LAST_AIM_POINT = (800, 400)
MISSILE_SPEED_MPS = 0.0

# Симуляция цели
TARGET_X, TARGET_Y = 800, 400
TARGET_SIZE = 60
MOVE_SPEED = 5
MOVE_INTERVAL = 16

key_state = {"Up": False, "Down": False, "Left": False, "Right": False}

# Окна
sim_window = None
canvas = None

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

        canv = tk.Canvas(overlay_window, width=size, height=size, bg='white', highlightthickness=0)
        canv.pack()
        if filled:
            canv.create_oval(0, 0, size, size, fill="red", outline="")
        else:
            canv.create_oval(5, 5, size-5, size-5, outline="red", width=3, fill="")
        overlay_window.update_idletasks()
    else:
        overlay_window.geometry(f"+{x-offset}+{y-offset}")
        canv = overlay_window.winfo_children()[0]
        canv.delete("all")
        if filled:
            canv.create_oval(0, 0, size, size, fill="red", outline="")
        else:
            canv.create_oval(5, 5, size-5, size-5, outline="red", width=3, fill="")
        overlay_window.update_idletasks()

# === Рисование ===
def draw_all():
    if not canvas:
        return
    canvas.delete("all")
    canvas.config(bg="gray")

    # Таймер и статус (левый верх)
    if RUNNING:
        elapsed = time.time() - START_TIME
        remaining = max(0.0, TOTAL_TIME - elapsed)
        dist_m = remaining / TOTAL_TIME * 1000 if TOTAL_TIME > 0 else 0
        status_text = f"До взрыва: {remaining:.1f} с\nРасст: {dist_m:.0f} м"
        canvas.create_text(10, 10, anchor="nw", text=status_text,
                           fill="white", font=("Courier", 10), tags="timer")

        # Статус цели
        try:
            sct = mss.mss()
            monitor = sct.monitors[1]
            screenshot = np.array(sct.grab(monitor))
            target_pos = find_triangle(screenshot)
        except Exception:
            target_pos = None
        status_color = "green" if target_pos else "red"
        status_str = "Цель: захвачена" if target_pos else "Цель: не найдена"
        canvas.create_text(10, 40, anchor="nw", text=status_str,
                           fill=status_color, font=("Arial", 10), tags="status")
    else:
        canvas.create_text(10, 10, anchor="nw", text="ВЗРЫВ!",
                           fill="red", font=("Arial", 16, "bold"), tags="timer")

    # Треугольник
    h = TARGET_SIZE * np.sqrt(3) / 2
    pts = [
        (TARGET_X, TARGET_Y - TARGET_SIZE // 2),
        (TARGET_X - TARGET_SIZE // 2, TARGET_Y + h // 2),
        (TARGET_X + TARGET_SIZE // 2, TARGET_Y + h // 2)
    ]
    canvas.create_polygon(pts, fill="black", outline="black", tags="triangle")

# === Клавиши ===
def on_key_press(event):
    if event.keysym in key_state:
        key_state[event.keysym] = True

def on_key_release(event):
    if event.keysym in key_state:
        key_state[event.keysym] = False

# === Цикл обновления ===
def update_loop():
    global TARGET_X, TARGET_Y, RUNNING

    if not RUNNING:
        draw_all()
        sim_window.after(100, update_loop)
        return

    # Движение только при зажатых клавишах
    dx = dy = 0
    if key_state["Up"]:    dy -= MOVE_SPEED
    if key_state["Down"]:  dy += MOVE_SPEED
    if key_state["Left"]:  dx -= MOVE_SPEED
    if key_state["Right"]: dx += MOVE_SPEED

    TARGET_X += dx
    TARGET_Y += dy

    draw_all()

    # Обновление наведения раз в секунду
    current_time = time.time()
    if not hasattr(update_loop, 'last_update'):
        update_loop.last_update = current_time
    if current_time - update_loop.last_update >= 0.05:
        elapsed = current_time - START_TIME
        remaining = max(0.0, TOTAL_TIME - elapsed)

        if remaining <= 0:
            # 💥 ВЗРЫВ
            update_overlay(TARGET_X, TARGET_Y, filled=True)
            RUNNING = False
            sim_window.unbind("<KeyPress>")
            sim_window.unbind("<KeyRelease>")
            draw_all()
            sim_window.after(100, update_loop)
            return

        # Поиск цели
        try:
            sct = mss.mss()
            monitor = sct.monitors[1]
            screenshot = np.array(sct.grab(monitor))
            target_pos = find_triangle(screenshot)
        except Exception:
            target_pos = None

        if target_pos is not None:
            target_history.append(target_pos)
            time_stamps.append(current_time)
        else:
            target_pos = (TARGET_X, TARGET_Y)  # fallback

        # Прогноз
        aim_point = target_pos
        if target_pos and len(target_history) >= 2:
            dt_total = time_stamps[-1] - time_stamps[0]
            if dt_total > 0:
                dx_t = target_history[-1][0] - target_history[0][0]
                dy_t = target_history[-1][1] - target_history[0][1]
                vx, vy = dx_t / dt_total, dy_t / dt_total
                pred_time = min(5.0, remaining)
                aim_x = int(target_pos[0] + vx * pred_time)
                aim_y = int(target_pos[1] + vy * pred_time)
                aim_point = (aim_x, aim_y)

        if aim_point:
            LAST_AIM_POINT = aim_point
            update_overlay(aim_point[0], aim_point[1], filled=False)

        update_loop.last_update = current_time

    sim_window.after(6, update_loop)

# === Стартовое окно ===
def show_start_form():
    root = tk.Tk()
    root.title("Запуск ракеты")
    root.geometry("300x180")
    root.resizable(False, False)
    root.wm_attributes("-topmost", True)

    tk.Label(root, text="Расстояние до цели (м):").pack(pady=5)
    dist_entry = tk.Entry(root, justify='center')
    dist_entry.insert(0, "1000")
    dist_entry.pack()

    tk.Label(root, text="Скорость припаса (км/ч):").pack(pady=5)
    speed_entry = tk.Entry(root, justify='center')
    speed_entry.insert(0, "400")
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

        # Создаём окно симуляции
        global sim_window, canvas
        sim_window = tk.Tk()
        sim_window.title("Наведение — цель")
        sim_window.attributes("-fullscreen", True)
        sim_window.configure(bg="gray")

        canvas = tk.Canvas(sim_window, bg="gray", highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        # Привязка клавиш
        sim_window.bind("<KeyPress>", on_key_press)
        sim_window.bind("<KeyRelease>", on_key_release)

        # Esc для выхода (только если не запущено)
        def on_escape(e):
            if not RUNNING:
                sim_window.destroy()
        sim_window.bind("<Escape>", on_escape)

        draw_all()
        sim_window.after(100, update_loop)
        sim_window.mainloop()

    tk.Button(root, text="🚀 ПУСК", command=on_launch, bg="lightgreen", font=("Arial", 12)).pack(pady=15)
    root.mainloop()

# === Запуск ===
if __name__ == "__main__":
    show_start_form()