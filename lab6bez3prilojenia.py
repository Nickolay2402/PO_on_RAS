import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 📐 ОБЩИЕ ПАРАМЕТРЫ (согласно методичке)
# =============================================================================
ampl = 3          # амплитуда сигнала
f0 = 10           # частота исходного сигнала
min_limit_x = 0   # начало оси времени
max_limit_x = 1   # конец оси времени
N = 512           # количество отсчётов
fd = N / max_limit_x  # частота дискретизации
t = np.linspace(min_limit_x, max_limit_x, N, endpoint=False)  # временная ось
M = 2 * N

# Частотная ось (надежный способ через numpy вместо ручного цикла из методички)
f_range = np.fft.fftshift(np.fft.fftfreq(M*2, d=1/fd))

# =============================================================================
# 📡 ПРИЛОЖЕНИЕ 1: ППРЧ (FHSS) - Псевдослучайная перестройка рабочей частоты
# =============================================================================
print("🔄 Приложение 1: ППРЧ...")
mas_psp4 = [1, 7, 9, 2, 8, 6, 4, 5, 3, 1]

casual_signal = []
signal_prp4 = []
for i in t:
    tmp = int(i // 0.1) % len(mas_psp4)  # % защищает от выхода за границы массива
    casual_signal.append(ampl * np.sin(2 * np.pi * f0 * i))
    signal_prp4.append(ampl * np.sin(2 * np.pi * f0 * mas_psp4[tmp] * i))

plt.figure(1, figsize=(10, 4))
plt.plot(t, signal_prp4, label='Сигнал с ППРЧ')
plt.plot(t, casual_signal, '--', alpha=0.7, label='Обычный сигнал')
plt.title('Временные реализации (ППРЧ)')
plt.xlabel('Время, с'); plt.ylabel('Амплитуда')
plt.grid(True); plt.legend()

# Спектры
fft_casual = np.fft.fftshift(np.fft.fft(casual_signal, M*2)) / N
fft_prp4   = np.fft.fftshift(np.fft.fft(signal_prp4, M*2)) / N

plt.figure(2, figsize=(10, 4))
plt.plot(f_range, np.abs(fft_casual), label='Обычный')
plt.plot(f_range, np.abs(fft_prp4), label='ППРЧ')
plt.title('Амплитудные спектры (ППРЧ)')
plt.xlabel('Частота, Гц'); plt.ylabel('Амплитуда')
plt.grid(True); plt.legend()

# =============================================================================
# 📡 ПРИЛОЖЕНИЕ 2: Цифровые манипуляции (АМн, ФМн)
# =============================================================================
print("📡 Приложение 2: Манипуляции...")
period = 0.05
mas_manip = [1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 
             -1, 1, -1, 1, -1, 1, -1, 1, 1, 1]
f1 = 40  # частота несущей

casual_sig2 = []
input_sig   = []
ampl_manip  = []
phas_manip  = []

for i in t:
    tmp = int(i // period) % len(mas_manip)
    bit = mas_manip[tmp]
    
    casual_sig2.append(ampl * np.sin(2 * np.pi * f1 * i))
    input_sig.append(bit)
    
    # Амплитудная манипуляция
    ampl_manip.append((ampl + bit) * np.sin(2 * np.pi * f1 * i))
    
    # Фазовая манипуляция (сдвиг на 0 или π)
    phase = 0 if bit == 1 else np.pi
    phas_manip.append(ampl * np.sin(2 * np.pi * f1 * i + phase))

plt.figure(3, figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(t, input_sig, 'k-', linewidth=1.5)
plt.title('Информационный сигнал')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, ampl_manip, label='АМн')
plt.plot(t, casual_sig2, '--', alpha=0.5, label='Несущая')
plt.title('Амплитудная манипуляция')
plt.legend(); plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, phas_manip, label='ФМн')
plt.plot(t, casual_sig2, '--', alpha=0.5, label='Несущая')
plt.title('Фазовая манипуляция')
plt.xlabel('Время, с'); plt.legend(); plt.grid(True)
plt.tight_layout()

# Спектры
fft_casual2 = np.fft.fftshift(np.fft.fft(casual_sig2, M*2)) / N
fft_phas    = np.fft.fftshift(np.fft.fft(phas_manip, M*2)) / N

plt.figure(4, figsize=(10, 4))
plt.plot(f_range, np.abs(fft_casual2), label='Обычный')
plt.plot(f_range, np.abs(fft_phas), label='ФМн')
plt.title('Амплитудные спектры (Манипуляции)')
plt.xlabel('Частота, Гц'); plt.ylabel('Амплитуда')
plt.grid(True); plt.legend()

# =============================================================================
# ✅ ВЫВОД ГРАФИКОВ
# =============================================================================
plt.tight_layout()
plt.show()  # Откроет все 4 окна одновременно (Figure 1-4)

print("\n✅ Все графики построены успешно!")
print("📊 Откройте окна Figure 1–4 для анализа результатов.")
print("💡 Приложение 3 (Код Баркера) исключено по вашему запросу.")