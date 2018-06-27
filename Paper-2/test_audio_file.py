# Test whether an Emergency signal is present in an audio sample

from em_detection import *

test_file = 'data/eval/emergency/17.wav'
y, sr = librosa.load(test_file, sr=8000)

classes = predict_probability(y, scaler)

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)
ax.plot(classes, c='r', linewidth = 3.0, alpha=0.5)
scale_x = len(classes)/10.0
ticks_x = ticker.FuncFormatter(lambda x, pos: '%.2f'%(x/scale_x))
ax.xaxis.set_major_formatter(ticks_x)
ax.set_yticks([0,1])
ax.set_ylim([-0.1, 1.1])
ax.set_xlabel("time (s)")
ax.set_ylabel("Em signal presence")
plt.grid('on')
plt.tight_layout()
plt.show()
