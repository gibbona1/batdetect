#do cnn on audio file using trained model

getwd()
setwd('C:/Users/Anthony/Documents/GitHub/batdetect/bat_train')

test_file <- 'data/labelled_data/uk/test/55ff08f49048f31f7800453c.wav'

library(tuneR)

test_audio <- readWave(test_file)

test_audio

play(test_audio)

library(ggplot2)
library(seewave)

?ggspectro
## first layer

#win_len <- as.integer(0.02322*test_audio@samp.rate)

#v <- ggspectro(test_audio, ovlp = 0.75, wl = win_len)

## using geom_tile ##
#v + geom_tile(aes(fill = amplitude)) + stat_contour()

#plot(test_audio)

#spec <- spectro(test_audio)#, ovlp = 0.75, wl = win_len)

# extract signal
snd = test_audio@left

# determine duration
dur = length(snd)/test_audio@samp.rate
dur # seconds
## [1] 3.840023

# determine sample rate
fs = test_audio@samp.rate
fs # Hz

snd = snd - mean(snd)

# plot waveform
plot(snd, type = 'l', xlab = 'Samples', ylab = 'Amplitude')
#dev.off()

reticulate::source_python("data_set_params.py")

params <- DataSetParams()

params$aug_shift

# number of points to use for the fft
nfft=1024

# window size (in points)
window=as.integer(params$fft_win_length*fs)

# overlap (in points)
overlap = window*params$fft_overlap

