# Deep-Learning Phase-Onset Picker for Deep Earth Seismology: PKIKP Waves
This is the repo for article: Deep-Learning Phase-Onset Picker for Deep Earth Seismology: PKIKP Waves published on Journal of Geophysical Research: Solid Earth.

## Content
- Trained model;
- Training data sets;
- Sliding-window picker;
- Demo notebook;
- PKIKP waveform examples;

## Recommended Pre-processing
- Original data: 150 s length, PKIKP onset predicted by ak135 located at 60 s
- Format: .sac
- Channel: Vertical channel
- Waveform length input to the picker: 50 s around ak135 prediction
- Sampling rate: 40 Hz
- Frequency filtering: 0.5 - 2 Hz

## Reference
Zhou, J., Phạm, T.‐S., & Tkalčić, H. (2024). Deep‐learning phase‐onset picker for deep Earth seismology: PKIKP waves. Journal of Geophysical Research: Solid Earth, 129, e2024JB029360. https://doi.org/10.1029/2024JB029360

