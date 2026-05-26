import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# ⚙️ НАСТРОЙКИ ВАРИАНТА (вставьте данные из таблицы ЛР №1 для вашей бригады)
# =============================================================================
VARIANT_PARAMS = {
    'f1': 20,      'A1': 1.0,   # Частота и амплитуда 1-го сигнала
    'f2': 25,      'A2': 0.8,   # Частота и амплитуда 2-го сигнала
    'f3': 30,      'A3': 0.5,   # Частота и амплитуда 3-го сигнала (если предусмотрено)
    'OSR_dB': 10,              # Отношение сигнал/шум (ОСШ) в децибелах
    'filter_f0': 25,           # Резонансная частота полосового фильтра
    'filter_bw': 10            # Полоса пропускания фильтра (Гц)
}

# =============================================================================
# 📐 ПАРАМЕТРЫ ДИСКРЕТИЗАЦИИ (согласно Приложению 1 методички)
# =============================================================================
N = 256                # Длина массива
T_max = 1.0            # Длительность реализации, с
fd = N / T_max         # Частота дискретизации, Гц
x = np.linspace(0, T_max, N, endpoint=False)  # Временная ось
M = 4 * N              # Длина массива для БПФ (интерполяция спектра)
f_range = np.fft.fftshift(np.fft.fftfreq(M, d=1/fd))  # Частотная ось для графиков

def plot_spectrum(signal, title):
    """Расчёт и построение амплитудного спектра по методике приложения 1"""
    fft_res = np.fft.fft(signal, M) / N
    fft_shifted = np.fft.fftshift(fft_res)
    amp_spectrum = np.abs(fft_shifted)
    
    plt.figure()
    plt.plot(f_range, amp_spectrum)
    plt.title(title)
    plt.xlabel('Частота, Гц')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.tight_layout()
    return fft_shifted  # Возвращаем комплексный спектр для дальнейшей обработки

# =============================================================================
# ✅ 5.1.1 Одиночный синусоидальный сигнал
# =============================================================================
sig1 = VARIANT_PARAMS['A1'] * np.sin(2 * np.pi * VARIANT_PARAMS['f1'] * x)

plt.figure()
plt.plot(x, sig1)
plt.title('5.1.1 Временная реализация одиночного сигнала')
plt.xlabel('Время, с')
plt.ylabel('Амплитуда')
plt.grid(True)

spec1_shifted = plot_spectrum(sig1, '5.1.1 Амплитудный спектр одиночного сигнала')

# =============================================================================
# ✅ 5.1.2 Сумма синусоидальных сигналов
# =============================================================================
sig2 = VARIANT_PARAMS['A2'] * np.sin(2 * np.pi * VARIANT_PARAMS['f2'] * x)
sig3 = VARIANT_PARAMS['A3'] * np.sin(2 * np.pi * VARIANT_PARAMS['f3'] * x)
sig_sum = sig1 + sig2 + sig3

plt.figure()
plt.plot(x, sig_sum)
plt.title('5.1.2 Временная реализация суммы сигналов')
plt.xlabel('Время, с')
plt.ylabel('Амплитуда')
plt.grid(True)

spec_sum_shifted = plot_spectrum(sig_sum, '5.1.2 Амплитудный спектр суммы сигналов')

# =============================================================================
# ✅ 5.1.3 Аддитивная смесь суммы сигналов и БГШ
# =============================================================================
# Расчёт мощности шума из заданного ОСШ (в дБ)
P_signal = np.mean(sig_sum**2)
OSR_linear = 10 ** (VARIANT_PARAMS['OSR_dB'] / 10)
P_noise = P_signal / OSR_linear
sigma_noise = np.sqrt(P_noise)

# Генерация Н(Б)ГШ
noise = np.random.normal(0, sigma_noise, N)
mixture = sig_sum + noise

plt.figure()
plt.plot(x, mixture)
plt.title('5.1.3 Временная реализация аддитивной смеси с БГШ')
plt.xlabel('Время, с')
plt.ylabel('Амплитуда')
plt.grid(True)

spec_mix_shifted = plot_spectrum(mixture, '5.1.3 Амплитудный спектр аддитивной смеси')

# =============================================================================
# ✅ 5.2 Построение прямоугольной АЧХ фильтра
# =============================================================================
a4h = np.zeros(M)
f0 = VARIANT_PARAMS['filter_f0']
bw = VARIANT_PARAMS['filter_bw']
f_min = f0 - bw / 2
f_max = f0 + bw / 2

# Заполняем АЧХ единицами в полосе пропускания (учитываем положительную и отрицательную частоты)
mask_pos = (f_range >= f_min) & (f_range <= f_max)
mask_neg = (f_range >= -f_max) & (f_range <= -f_min)
a4h[mask_pos | mask_neg] = 1.0

plt.figure()
plt.plot(f_range, a4h)
plt.title('5.2 АЧХ полосового фильтра (прямоугольная)')
plt.xlabel('Частота, Гц')
plt.ylabel('Коэффициент передачи')
plt.ylim(-0.1, 1.2)
plt.grid(True)

# =============================================================================
# ✅ 5.3 Фильтрация в спектральной области и обратное БПФ
# =============================================================================
# Перемножение спектра смеси и АЧХ фильтра
filtered_spec_shifted = spec_mix_shifted * a4h

# Возвращаем спектр в "не сдвинутый" вид для обратного преобразования
filtered_spec = np.fft.ifftshift(filtered_spec_shifted) * N  # *N для компенсации нормировки fft()

# Обратное БПФ
filtered_signal_full = np.fft.ifft(filtered_spec)
# Берём только первые N отсчётов (реальная часть)
filtered_signal = filtered_signal_full[:N].real

plt.figure()
plt.plot(x, filtered_signal)
plt.title('5.3 Временная реализация после фильтрации (обратное БПФ)')
plt.xlabel('Время, с')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.show()

# Вывод параметров для отчёта
print("\n📊 Параметры использованного варианта:")
print(f"Сигналы: f1={VARIANT_PARAMS['f1']}Гц, f2={VARIANT_PARAMS['f2']}Гц, f3={VARIANT_PARAMS['f3']}Гц")
print(f"ОСШ: {VARIANT_PARAMS['OSR_dB']} дБ")
print(f"Фильтр: f_рез={VARIANT_PARAMS['filter_f0']}Гц, Δf={VARIANT_PARAMS['filter_bw']}Гц")
print("✅ Все графики построены. Данные готовы для вставки в отчёт.")

