function run_live_recording_pipeline(outputWav)

if nargin < 1 || isempty(outputWav)
    outputWav = "live_processed.wav";
end

thisFile = mfilename('fullpath');
thisDir  = fileparts(thisFile);
addpath(fullfile(thisDir, "separateSpeakers"));

%% CONFIG 
fs_rec_expected = 16000;     % expected phone sampling rate (we buffer at this)
fs_proc = 16000;             % processing sampling rate (stream)
chunk_ms = 40;               % target chunk processing granularity (ms)
chunk_samples_proc = round(fs_proc * (chunk_ms/1000)); 

% STFT params (at fs_proc)
nfft = 512;
winlen = 512;    
hop = 128;        
win = hann(winlen, 'periodic');
epsv = 1e-12;

% MVDR streaming params
d_mic = 0.12;          % meters
c = 340;
steerAzDeg = 90;
tau_sec = 0.5;         % covariance time constant for EMA (0.5 s)
frame_time = hop / fs_proc;
alpha = exp(-frame_time / tau_sec); % EMA forgetting factor per frame

% Safety / regularization
reg_eps = 1e-3;

%%  INIT mobiledev 
m = mobiledev;
if ~isprop(m,'Logging')
    error('mobiledev object does not appear to work on this MATLAB instance.');
end
if isprop(m,'Connected') && ~m.Connected
    warning("mobiledev reports Connected = 0. Ensure MATLAB Mobile is paired and on same network.");
end

fprintf("Waiting for MATLAB Mobile to start recording (press Record in the phone app)...\n");

% Wait until recording starts. This avoids the internal timestamp bug.
waitTimeout = 30; % seconds, increased to be safe for demos
t0 = tic;
while ~m.Logging
    if toc(t0) > waitTimeout
        error("Timeout waiting for mobile device to start recording. Check MATLAB Mobile connection.");
    end
    pause(0.12);
end
fprintf("Recording detected. Starting stream processing. (Stop recording on phone to finish.)\n");

%% TIMING CONTAINERS 
readTimes = [];        % time spent in readAudio calls
resampleTimes = [];    % time spent resampling chunks
procTimes = [];        % total time per processed frame (winlen -> hop emit)
stftTimes = [];        % STFT/FFT time
covTimes = [];         % covariance + weight compute (MVDR) time
istftTimes = [];       % ISTFT + OLA time

% counters
totalChunks = 0;
totalSamplesProcessed = 0;

%% INIT BUFFERS & CONSTANTS 
rawBuffer = zeros(0,2);      % raw at fs_rec_expected
procBuffer = zeros(0,2);     % resampled buffer at fs_proc
outOLA = [];                 % output samples (at fs_proc)
olaBuffer = zeros(winlen,1); % rolling synthesis buffer

halfBins = nfft/2 + 1;
freqs = (0:halfBins-1)' * fs_proc / nfft;
az = deg2rad(steerAzDeg);
d = (d_mic) * [1;0]; % mic spacing along x
u = [cos(az); sin(az)];
td = -dot(d, u) / c;
A = [ones(halfBins,1), exp(-1j*2*pi*freqs*td)]; % steering (halfBins x 2)

% Initialize streaming covariance and beamformer
R = repmat(eye(2) * 1e-6, [1,1,halfBins]); % small initial covariance
W = zeros(2, halfBins);

% mapping between record-sample-chunk and proc-sample-chunk
chunk_samples_rec = round(chunk_samples_proc * (fs_rec_expected / fs_proc));

lastNumRaw = 0; % used to diff incoming mobile buffer

%% MAIN POLLING + PROCESSING LOOP 
while m.Logging
    % read audio 
    try
        t_read = tic;
        audTT = readAudio(m, OutputFormat="timetable");
        readTimes(end+1) = toc(t_read);
    catch ME
        warning("readAudio transient failure: %s. Retrying...", ME.message);
        pause(0.06);
        continue;
    end

    % Extract samples 
    if istimetable(audTT)
        audData = audTT.Variables;
    elseif isnumeric(audTT)
        audData = audTT;
    else
        warning("readAudio returned unexpected format; skipping this read.");
        pause(0.06);
        continue;
    end

    
    if isempty(audData)
        pause(0.06);
        continue;
    end
    if size(audData,2) == 1
        error("Mobile recording appears mono. Need Nx2 two-channel recording from phone.");
    elseif size(audData,2) > 2
        audData = audData(:,1:2);
    end

    % Append only new samples (mobiledev returns entire buffer each call)
    if size(audData,1) > lastNumRaw
        newSeg = audData(lastNumRaw+1:end, :);
        rawBuffer = [rawBuffer; newSeg];
        lastNumRaw = size(audData,1);
    end

    % Process available rawBuffer in record-chunks
    while size(rawBuffer,1) >= chunk_samples_rec
        % Pop chunk in recording fs
        chunk_rec = rawBuffer(1:chunk_samples_rec, :);
        rawBuffer(1:chunk_samples_rec, :) = [];
        lastNumRaw = lastNumRaw - chunk_samples_rec;

        % Resample chunk to processing rate 
        t_res = tic;
        x1_proc = resample(chunk_rec(:,1), fs_proc, fs_rec_expected);
        x2_proc = resample(chunk_rec(:,2), fs_proc, fs_rec_expected);
        resampleTimes(end+1) = toc(t_res);
        chunk_proc = [x1_proc, x2_proc];
        procBuffer = [procBuffer; chunk_proc];

        % Process procBuffer hop-by-hop
        while size(procBuffer,1) >= winlen
            totalChunks = totalChunks + 1;
            t_chunk = tic;  % whole frame timer

            % STFT / frame assembly 
            t = tic;
            frame = procBuffer(1:winlen, :);
            procBuffer(1:hop, :) = []; % consume hop
            frameWin = frame .* win;
            % FFT per channel
            X1f = fft(frameWin(:,1), nfft);
            X2f = fft(frameWin(:,2), nfft);
            Xf = [X1f(1:halfBins), X2f(1:halfBins)];
            stftTimes(end+1) = toc(t);

            % Covariance update + MVDR weight compute 
            t = tic;
            Yk = zeros(halfBins,1);
            for k = 1:halfBins
                xk = Xf(k,:).';
                Rinst = (xk * xk');
                R(:,:,k) = alpha * R(:,:,k) + (1-alpha) * Rinst;
                Rreg = R(:,:,k) + eye(2) * (trace(R(:,:,k)) * reg_eps + epsv);
                ak = A(k,:).';
                denom = (ak' * (Rreg \ ak)) + epsv;
                w = (Rreg \ ak) / denom;
                W(:,k) = w;
                Yk(k) = w' * xk;
            end
            covTimes(end+1) = toc(t);

            % ISTFT / overlap-add 
            t = tic;
            Yfull = [Yk; conj(Yk(end-1:-1:2))];
            yframe = real(ifft(Yfull, nfft));
            yframe = yframe(1:winlen) .* win;

            % Overlap-add synthesis using rolling olaBuffer
            olaBuffer = olaBuffer + yframe;
            emit = olaBuffer(1:hop);
            outOLA = [outOLA; emit];
            olaBuffer(1:end-hop) = olaBuffer(hop+1:end);
            olaBuffer(end-hop+1:end) = 0;
            istftTimes(end+1) = toc(t);

            % finalize chunk
            procTimes(end+1) = toc(t_chunk);
            totalSamplesProcessed = totalSamplesProcessed + hop; % hop samples emitted per frame
        end
    end

    pause(0.04); % avoid busy-wait
end

%% After recording stops
fprintf("Recording stopped on phone. Finalizing (flushing buffers)...\n");

% Process any remaining procBuffer frames
if ~isempty(procBuffer)
    padLen = max(0, winlen - size(procBuffer,1));
    procBuffer = [procBuffer; zeros(padLen,2)];
end

while size(procBuffer,1) >= winlen
    t_chunk = tic;
    frame = procBuffer(1:winlen,:);
    procBuffer(1:hop,:) = [];
    frameWin = frame .* win;

    X1f = fft(frameWin(:,1), nfft);
    X2f = fft(frameWin(:,2), nfft);
    Xf = [X1f(1:halfBins), X2f(1:halfBins)];

    t = tic;
    Yk = zeros(halfBins,1);
    for k = 1:halfBins
        xk = Xf(k,:).';
        Rinst = (xk * xk');
        R(:,:,k) = alpha * R(:,:,k) + (1-alpha) * Rinst;
        Rreg = R(:,:,k) + eye(2) * (trace(R(:,:,k)) * reg_eps + epsv);
        ak = A(k,:).';
        denom = (ak' * (Rreg \ ak)) + epsv;
        w = (Rreg \ ak) / denom;
        W(:,k) = w;
        Yk(k) = w' * xk;
    end
    covTimes(end+1) = toc(t);

    Yfull = [Yk; conj(Yk(end-1:-1:2))];
    yframe = real(ifft(Yfull, nfft));
    yframe = yframe(1:winlen) .* win;
    olaBuffer = olaBuffer + yframe;
    emit = olaBuffer(1:hop);
    outOLA = [outOLA; emit];
    olaBuffer(1:end-hop) = olaBuffer(hop+1:end);
    olaBuffer(end-hop+1:end) = 0;
    istftTimes(end+1) = 0; % already included in loop above (no separate timer here)
    procTimes(end+1) = toc(t_chunk);
    totalSamplesProcessed = totalSamplesProcessed + hop;
end

% append any nonzero tail of olaBuffer
tailNonzero = find(abs(olaBuffer) > 0, 1, 'last');
if ~isempty(tailNonzero)
    outOLA = [outOLA; olaBuffer(1:tailNonzero)];
end

% final beamformed output at fs_proc
y_bf = outOLA(:);

%% OFFLINE POSTPROCESSING
t_offline = tic;
if exist('separateSpeakers','file') == 2
    try
        fprintf("Applying separateSpeakers (offline)...\n");
        y_sep = separateSpeakers(y_bf, fs_proc, 'NumSpeakers', 1);
        y_bf = y_sep(:);
    catch ME
        warning("separateSpeakers failed during offline: %s. Skipping.", ME.message);
    end
else
    fprintf("separateSpeakers not found — skipping offline separation.\n");
end

if exist('enhanceSpeech','file') == 2
    try
        fprintf("Applying enhanceSpeech (offline)...\n");
        y_es = enhanceSpeech(y_bf, fs_proc);
        rms_ref = sqrt(mean(y_bf.^2)+epsv);
        rms_1 = sqrt(mean(y_es.^2)+epsv);
        y_bf = 0.98 * y_es * (rms_ref / rms_1) / max(abs(y_es)+epsv);
    catch ME
        warning("enhanceSpeech failed during offline: %s. Skipping.", ME.message);
    end
else
    fprintf("enhanceSpeech not found — skipping offline enhancement.\n");
end
offlineTime = toc(t_offline);

%% STATS & RTF 
% Use the produced output length for the audio duration (robust)
totalAudioSec = numel(y_bf) / fs_proc;

% total measured processing time (streaming core)
totalStreamingTime = sum(readTimes) + sum(resampleTimes) + sum(procTimes);

% helper for stats
statfun = @(v) struct('n', numel(v), 'mean', mean(v), 'median', median(v), 'std', std(v), 'p95', prctile(v,95));

readS = []; resampS = []; procS = []; stftS = []; covS = []; istftS = [];
if ~isempty(readTimes), readS = statfun(readTimes); end
if ~isempty(resampleTimes), resampS = statfun(resampleTimes); end
if ~isempty(procTimes), procS = statfun(procTimes); end
if ~isempty(stftTimes), stftS = statfun(stftTimes); end
if ~isempty(covTimes), covS = statfun(covTimes); end
if ~isempty(istftTimes), istftS = statfun(istftTimes); end

RTF_stream = totalStreamingTime / totalAudioSec;
RTF_offline = offlineTime / totalAudioSec;

fprintf("\n===== Timing summary =====\n");
fprintf("Total output duration: %.3f s (samples=%d)\n", totalAudioSec, numel(y_bf));
fprintf("Streaming core total time (read+resample+proc): %.3f s  --> RTF_stream = %.3f\n", totalStreamingTime, RTF_stream);
fprintf("Offline neural postproc time: %.3f s  --> RTF_offline = %.3f\n", offlineTime, RTF_offline);
fprintf("Chunks processed: %d\n\n", totalChunks);

if ~isempty(procS)
    fprintf("Per-frame (winlen->hop) processing median = %.3f ms, mean = %.3f ms, p95 = %.3f ms (N=%d)\n", ...
        1000*procS.median, 1000*procS.mean, 1000*procS.p95, procS.n);
end
if ~isempty(stftS)
    fprintf("  STFT/FFT median = %.3f ms\n", 1000*stftS.median);
end
if ~isempty(covS)
    fprintf("  MVDR cov+weights median = %.3f ms\n", 1000*covS.median);
end
if ~isempty(istftS)
    fprintf("  ISTFT+OLA median = %.3f ms\n", 1000*istftS.median);
end
if ~isempty(readS)
    fprintf("readAudio median = %.3f ms (mean=%.3f ms, N=%d)\n", 1000*readS.median, 1000*readS.mean, readS.n);
end
if ~isempty(resampS)
    fprintf("resample median = %.3f ms (mean=%.3f ms, N=%d)\n", 1000*resampS.median, 1000*resampS.mean, resampS.n);
end

fprintf("\nSaved processed audio to %s (fs=%d)\n", outputWav, fs_proc);

%% WRITE OUTPUT
% Final normalization & write
y_norm = 0.98 * y_bf / max(abs(y_bf) + epsv);
audiowrite(outputWav, y_norm, fs_proc);

end