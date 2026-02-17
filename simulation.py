import cv2
import numpy as np
import mss
import time
import tkinter as tk
from collections import deque
import math

# === Глобальные переменные ===
target_history = deque(maxlen=10)
time_stamps = deque(maxlen=10)
overlay_window = None  # красный кружок (упреждение)
missile_window = None  # жёлтый кружок (ракета)
explosion_window = None  # красный взрыв (если будет)
overlay_frozen = False  # флаг заморозки красного кружка
missile_frozen = False  # флаг заморозки жёлтого кружка

# Параметры полёта
TOTAL_TIME = 0.0
START_TIME = 0.0
RUNNING = False
LAST_AIM_POINT = (800, 400)
MISSILE_SPEED_MPS = 0.0

# Симуляция цели
TARGET_X, TARGET_Y = 800, 400
TARGET_SIZE = 60
MOVE_SPEED = 4
UPDATE_INTERVAL_MS = 8

key_state = {"Up": False, "Down": False, "Left": False, "Right": False}

# Окна
sim_window = None
canvas = None

# === Физика ракеты ===
MISSILE_X, MISSILE_Y = 600, 400  # стартовая позиция ракеты
MISSILE_VX, MISSILE_VY = 0, 0  # скорость ракеты
MISSILE_SPEED = 0.0  # текущая скорость

# Ограничения ракеты (ОЧЕНЬ АГРЕССИВНЫЕ)
MAX_MISSILE_SPEED = 400.0  # px/сек (высокая)
MAX_ACCELERATION = 300.0  # px/сек² (высокое ускорение)
MAX_DECELERATION = 400.0  # px/сек² (очень высокое торможение)
MISSILE_RADIUS = 12  # радиус жёлтого кружка
OVERLAY_RADIUS = 15  # радиус красного кружка

# Параметры наведения (АГРЕССИВНЫЕ)
AGGRESSIVENESS = 1.5  # высокая агрессивность
FINAL_APPROACH_TIME = 1.5  # последние 1.5 сек — фиксированное упреждение

last_frame_time = time.perf_counter()


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


# === Обновление оверлея упреждения (красный контур, 30 пикселей) ===
def update_overlay(x, y):
    global overlay_window
    size = OVERLAY_RADIUS * 2
    offset = size // 2

    if overlay_window is None:
        overlay_window = tk.Tk()
        overlay_window.overrideredirect(True)
        overlay_window.wm_attributes("-topmost", True)
        overlay_window.wm_attributes("-transparentcolor", "white")
        overlay_window.geometry(f"{size}x{size}+{x - offset}+{y - offset}")

        canv = tk.Canvas(overlay_window, width=size, height=size, bg='white', highlightthickness=0)
        canv.pack()
        canv.create_oval(3, 3, size - 3, size - 3, outline="red", width=2, fill="")
        overlay_window.update_idletasks()
    else:
        overlay_window.geometry(f"+{x - offset}+{y - offset}")
        overlay_window.update_idletasks()


# === Обновление ракеты (жёлтый контур → жёлтый закрашенный при взрыве) ===
def update_missile(x, y, exploded=False):
    global missile_window
    size = MISSILE_RADIUS * 2
    offset = size // 2

    if missile_window is None:
        missile_window = tk.Tk()
        missile_window.overrideredirect(True)
        missile_window.wm_attributes("-topmost", True)
        missile_window.wm_attributes("-transparentcolor", "white")
        missile_window.geometry(f"{size}x{size}+{x - offset}+{y - offset}")

        canv = tk.Canvas(missile_window, width=size, height=size, bg='white', highlightthickness=0)
        canv.pack()
        if exploded:
            canv.create_oval(0, 0, size, size, fill="yellow", outline="orange", width=2)
        else:
            canv.create_oval(2, 2, size - 2, size - 2, outline="yellow", width=2, fill="")
        missile_window.update_idletasks()
    else:
        missile_window.geometry(f"+{x - offset}+{y - offset}")
        canv = missile_window.winfo_children()[0]
        canv.delete("all")
        if exploded:
            canv.create_oval(0, 0, size, size, fill="yellow", outline="orange", width=2)
        else:
            canv.create_oval(2, 2, size - 2, size - 2, outline="yellow", width=2, fill="")
        missile_window.update_idletasks()


# === Симуляция полёта ракеты (ОЧЕНЬ АГРЕССИВНАЯ) ===
def simulate_missile(aim_x, aim_y, dt):
    global MISSILE_X, MISSILE_Y, MISSILE_VX, MISSILE_VY, MISSILE_SPEED

    dx = aim_x - MISSILE_X
    dy = aim_y - MISSILE_Y
    distance = math.hypot(dx, dy)

    if distance < 1.0:
        MISSILE_VX *= 0.95
        MISSILE_VY *= 0.95
        return

    # ОЧЕНЬ АГРЕССИВНОЕ наведение: максимальная скорость почти сразу
    desired_speed = min(MAX_MISSILE_SPEED, distance * AGGRESSIVENESS)

    dir_x = dx / distance
    dir_y = dy / distance
    desired_vx = dir_x * desired_speed
    desired_vy = dir_y * desired_speed

    current_speed = math.hypot(MISSILE_VX, MISSILE_VY)
    desired_speed_mag = math.hypot(desired_vx, desired_vy)

    # Высокое ускорение/торможение
    if desired_speed_mag > current_speed:
        max_accel_this_frame = MAX_ACCELERATION * dt
    else:
        max_accel_this_frame = MAX_DECELERATION * dt

    dvx = desired_vx - MISSILE_VX
    dvy = desired_vy - MISSILE_VY
    dv = math.hypot(dvx, dvy)

    if dv > 0:
        accel = min(dv, max_accel_this_frame)
        MISSILE_VX += (dvx / dv) * accel
        MISSILE_VY += (dvy / dv) * accel

    # Ограничение максимальной скорости
    current_speed = math.hypot(MISSILE_VX, MISSILE_VY)
    if current_speed > MAX_MISSILE_SPEED:
        MISSILE_VX = (MISSILE_VX / current_speed) * MAX_MISSILE_SPEED
        MISSILE_VY = (MISSILE_VY / current_speed) * MAX_MISSILE_SPEED

    MISSILE_X += MISSILE_VX * dt
    MISSILE_Y += MISSILE_VY * dt
    MISSILE_SPEED = math.hypot(MISSILE_VX, MISSILE_VY)


# === Рисование ===
def draw_all():
    if not canvas:
        return
    canvas.delete("all")
    canvas.config(bg="gray")

    if RUNNING:
        elapsed = time.time() - START_TIME
        remaining = max(0.0, TOTAL_TIME - elapsed)
        dist_m = remaining / TOTAL_TIME * 1000 if TOTAL_TIME > 0 else 0
        status_text = f"До взрыва: {remaining:.1f} с | Расст: {dist_m:.0f} м"
        canvas.create_text(10, 10, anchor="nw", text=status_text,
                           fill="white", font=("Courier", 10), tags="timer")

        try:
            sct = mss.mss()
            monitor = sct.monitors[1]
            screenshot = np.array(sct.grab(monitor))
            target_pos = find_triangle(screenshot)
        except Exception:
            target_pos = None
        status_color = "green" if target_pos else "red"
        status_str = "🎯 Цель: захвачена" if target_pos else "❌ Цель: не найдена"
        canvas.create_text(10, 30, anchor="nw", text=status_str,
                           fill=status_color, font=("Arial", 10), tags="status")

        distance_to_aim = math.hypot(MISSILE_X - LAST_AIM_POINT[0], MISSILE_Y - LAST_AIM_POINT[1])
        missile_status = f"🚀 Ракета: {MISSILE_SPEED:.0f} px/s | До цели: {distance_to_aim:.0f}px"
        canvas.create_text(10, 50, anchor="nw", text=missile_status,
                           fill="yellow", font=("Courier", 9), tags="missile_status")
    else:
        canvas.create_text(10, 10, anchor="nw", text="💥 ВЗРЫВ!",
                           fill="red", font=("Arial", 16, "bold"), tags="timer")

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


# === Перезапуск симуляции (полный перезапуск с окна ввода данных) ===
def restart_simulation():
    global sim_window, overlay_window, missile_window, explosion_window
    global RUNNING

    RUNNING = False

    # Уничтожаем все окна
    if sim_window:
        sim_window.destroy()
    if overlay_window:
        overlay_window.destroy()
    if missile_window:
        missile_window.destroy()
    if explosion_window:
        explosion_window.destroy()

    # Сбрасываем глобальные переменные окон
    sim_window = None
    overlay_window = None
    missile_window = None
    explosion_window = None

    # Открываем заново стартовое окно
    show_start_form()


# === Цикл обновления ===
def update_loop():
    global TARGET_X, TARGET_Y, RUNNING, last_frame_time, LAST_AIM_POINT
    global overlay_frozen, missile_frozen

    now = time.perf_counter()
    dt = now - last_frame_time
    last_frame_time = now

    if not RUNNING:
        draw_all()
        sim_window.after(UPDATE_INTERVAL_MS, update_loop)
        return

    dx = dy = 0
    if key_state["Up"]:    dy -= MOVE_SPEED
    if key_state["Down"]:  dy += MOVE_SPEED
    if key_state["Left"]:  dx -= MOVE_SPEED
    if key_state["Right"]: dx += MOVE_SPEED

    TARGET_X += dx
    TARGET_Y += dy

    draw_all()

    current_time = time.time()
    if not hasattr(update_loop, 'last_update'):
        update_loop.last_update = current_time

    if not overlay_frozen and current_time - update_loop.last_update >= 0.1:
        elapsed = current_time - START_TIME
        remaining = max(0.0, TOTAL_TIME - elapsed)

        if remaining <= 0 and not missile_frozen:
            # 💥 ВЗРЫВ
            overlay_frozen = True
            missile_frozen = True
            update_missile(int(MISSILE_X), int(MISSILE_Y), exploded=True)
            RUNNING = False
            sim_window.unbind("<KeyPress>")
            sim_window.unbind("<KeyRelease>")
            draw_all()
            sim_window.after(UPDATE_INTERVAL_MS, update_loop)
            return

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
            target_pos = (TARGET_X, TARGET_Y)

        aim_point = target_pos
        if target_pos and len(target_history) >= 2:
            dt_total = time_stamps[-1] - time_stamps[0]
            if dt_total > 0:
                dx_t = target_history[-1][0] - target_history[0][0]
                dy_t = target_history[-1][1] - target_history[0][1]
                vx, vy = dx_t / dt_total, dy_t / dt_total

                # 🔑 КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: последние 1.5 сек — фиксированное упреждение 1 сек
                if remaining > FINAL_APPROACH_TIME:
                    pred_time = min(5.0, remaining)
                else:
                    pred_time = 1.0  # фиксировано 1 сек в последние 1.5 сек

                aim_x = int(target_pos[0] + vx * pred_time)
                aim_y = int(target_pos[1] + vy * pred_time)
                aim_point = (aim_x, aim_y)

        if aim_point:
            LAST_AIM_POINT = aim_point
            if not overlay_frozen:
                update_overlay(aim_point[0], aim_point[1])

        update_loop.last_update = current_time

    if LAST_AIM_POINT is not None and not missile_frozen:
        simulate_missile(LAST_AIM_POINT[0], LAST_AIM_POINT[1], dt)
        update_missile(int(MISSILE_X), int(MISSILE_Y), exploded=False)

    sim_window.after(UPDATE_INTERVAL_MS, update_loop)


# === Стартовое окно ===
def show_start_form():
    root = tk.Tk()
    root.title("Запуск ракеты")
    root.geometry("350x220")
    root.resizable(False, False)
    root.wm_attributes("-topmost", True)

    tk.Label(root, text="Расстояние до цели (м):", font=("Arial", 10)).pack(pady=5)
    dist_entry = tk.Entry(root, justify='center', font=("Arial", 12))
    dist_entry.insert(0, "1000")
    dist_entry.pack()

    tk.Label(root, text="Скорость припаса (км/ч):", font=("Arial", 10)).pack(pady=5)
    speed_entry = tk.Entry(root, justify='center', font=("Arial", 12))
    speed_entry.insert(0, "300")
    speed_entry.pack()

    tk.Label(root, text="Начальная позиция ракеты (X, Y):", font=("Arial", 9)).pack(pady=5)
    pos_entry = tk.Entry(root, justify='center', font=("Arial", 10))
    pos_entry.insert(0, "600, 400")
    pos_entry.pack()

    def on_launch():
        try:
            distance_m = float(dist_entry.get())
            speed_kmh = float(speed_entry.get())
            pos_str = pos_entry.get().replace(" ", "")
            missile_x, missile_y = map(int, pos_str.split(","))
            if distance_m <= 0 or speed_kmh <= 0:
                raise ValueError
        except:
            tk.messagebox.showerror("Ошибка", "Введите корректные числа!")
            return

        global TOTAL_TIME, START_TIME, RUNNING, MISSILE_SPEED_MPS
        global MISSILE_X, MISSILE_Y, MISSILE_VX, MISSILE_VY, LAST_AIM_POINT
        global overlay_frozen, missile_frozen
        global sim_window, canvas

        MISSILE_SPEED_MPS = speed_kmh * 1000 / 3600
        TOTAL_TIME = distance_m / MISSILE_SPEED_MPS
        START_TIME = time.time()
        RUNNING = True
        overlay_frozen = False
        missile_frozen = False

        MISSILE_X, MISSILE_Y = missile_x, missile_y
        MISSILE_VX, MISSILE_VY = 0, 0
        LAST_AIM_POINT = (missile_x, missile_y)

        root.destroy()

        sim_window = tk.Tk()
        sim_window.title("Наведение — цель")
        sim_window.attributes("-fullscreen", True)
        sim_window.configure(bg="gray")

        canvas = tk.Canvas(sim_window, bg="gray", highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        # Кнопка перезапуска в правом верхнем углу
        restart_button = tk.Button(
            sim_window,
            text="↻ Перезапуск",
            command=restart_simulation,
            bg="lightblue",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5
        )
        restart_button.place(x=sim_window.winfo_screenwidth() - 150, y=10)

        sim_window.bind("<KeyPress>", on_key_press)
        sim_window.bind("<KeyRelease>", on_key_release)

        def on_escape(e):
            if not RUNNING:
                sim_window.destroy()

        sim_window.bind("<Escape>", on_escape)

        draw_all()
        sim_window.after(UPDATE_INTERVAL_MS, update_loop)
        sim_window.mainloop()

    tk.Button(root, text="🚀 ПУСК", command=on_launch, bg="lightgreen", font=("Arial", 14, "bold")).pack(pady=15)
    root.mainloop()


# === Запуск ===
if __name__ == "__main__":
    show_start_form()