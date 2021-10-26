#Convert of run_detector.py to R

read_baseline_res <- function(baseline_file_name, test_files){
  da   <- read.csv(baseline_file_name)
  pos  <- c()
  prob <- c()
  for(ff in test_files){
    rr   <- da[da['Filename'] == ff,]
    inds <- sort(rr$TimeInFile, index.return=TRUE)
    pos  <- c(pos,rr$TimeInFile[inds])
    prob <- c(prob,rr$Quality[inds])
  }
  return(list("pos"=pos, "prob"=prob))
}

read_audio <- function(file_name, do_time_expansion, chunk_size, win_size){
  
  # try to read in audio file
  audiolist <- trycatch(
    tuneR::readWave(file_name),
    error=function() {
      message('  Error reading file')
      return(list(read_fail      = TRUE, 
                  audio_pad      = NA, 
                  file_dur       = NA, 
                  samp_rate      = NA, 
                  samp_rate_orig = NA))
    }
  )
  samp_rate_orig <- audiolist$samp.rate
  
  audio <- audiolist$left

  # convert to mono if stereo
  if(is.null(audio$right))
    print('  Warning: stereo file. Just taking left channel.')
  
  file_dur <- length(audio) / float(samp_rate_orig)
  cat('  dur', round(file_dur,3), '(secs) , fs', samp_rate_orig)
  
  # original model is trained on time expanded data
  samp_rate <- samp_rate_orig
  if(do_time_expansion){
    samp_rate <- as.integer(samp_rate_orig/10.0)
    file_dur  <- file_dur*10
  }
  
  rep(0,)
  
  multiplier <- ceiling(file_dur/float(chunk_size-win_size))
  diff       <- multiplier*(chunk_size-win_size) - file_dur + win_size
  audio_pad  <- c(audio, rep(0, as.integer(diff*samp_rate)))
  
  return(list(read_fail = False, 
              audio_pad, 
              file_dur, 
              samp_rate, 
              samp_rate_orig))
}

run_detector <- function(det, audio, file_dur, samp_rate, detection_thresh, params){
    
  det_time <- c()
  det_prob <- c()
  
  # files can be long so we split each up into separate (overlapping) chunks
  st_positions <- seq.int(from=0, to=file_dur, by=(params$chunk_size-params$window_size))
  #print('st_positions',st_positions)
  for(chunk_id in seq_along(st_positions)){
    
    # take a chunk of the audio
    # should already be zero padded at the end so its the correct size
    st_pos      <- as.integer(st_positions[chunk_id]*samp_rate)
    en_pos      <- as.integer(st_pos + params$chunk_size*samp_rate)
    audio_chunk <- audio[st_pos:en_pos]
    # make predictions
    chunk_spec <- compute_features(audio_chunk, samp_rate, params)
    chunk_spec <- np.squeeze(chunk_spec)
    chunk_spec <- np.expand_dims(chunk_spec,-1)
    
    det_pred <- det$predict(chunk_spec)
  
    if(params$smooth_op_prediction){
      det_pred <- gaussian_filter1d(det_pred, params$smooth_op_prediction_sigma, axis=0)
    }
    
    pos, prob <- nms_1d(det_pred[:,0], params$nms_win_size, file_dur)
    #pos      = np.argmax(det_pred, axis=-1)
    prob = prob[:,0]
    #print('pos.shape', pos.shape)
    #print('prob.shape', prob.shape)
    #prob     = det_pred[:, 0]
    #print('(prob >= detection_thresh).shape', (prob >= detection_thresh).shape)
    #print('(pos < (params.chunk_size-(params.window_size/2.0)).shape',
    #(pos < (params.chunk_size-(params.window_size/2.0))).shape)
    #print((prob >= detection_thresh) & (pos < (params.chunk_size-(params.window_size/2.0))))
    #print(chunk_id)
    #print(len(st_positions)-1)
    # remove predictions near the end (if not last chunk) and ones that are
    # below the detection threshold
    if(chunk_id == len(st_positions))
      inds = (prob >= detection_thresh)
    else
      inds = (prob >= detection_thresh) & (pos < (params$chunk_size-(params$window_size/2.0)))
    
    # convert detection time back into global time and save valid detections
    #print('inds.shape', inds.shape)
    #print('inds', inds)
    #print('st_position',st_position)
    if pos.shape[0] > 0:
      det_time.append(pos[inds] + st_position)
    det_prob.append(prob[inds])
  }
  if len(det_time) > 0:
    det_time = np.hstack(det_time)
  det_prob = np.hstack(det_prob)
  
  # undo the effects of times expansion
  if do_time_expansion:
    det_time /= 10.0
  
  return det_time, det_prob
}
