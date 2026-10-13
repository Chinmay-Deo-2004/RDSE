clear;
close all;
clc;

%% CONFIG
load("dereverbNet.mat");
cfg.task = 2; % 1 and 2 pertain to an anechoic chamber and reverberant room resepctively
cfg.useDereverb = 0; %Note: toggle-able feature, 'de-reverb mode'

% artifact naming
artifact.taskTag = sprintf("task%d", cfg.task);
artifact.enhancedWav = sprintf( ... 
    'SPCup_%s.wav', artifact.taskTag);
artifact.refWav = sprintf( ... 
    'ref_%s.wav', artifact.taskTag);

% Ensure seperateSpeakers is added to path
thisFile = mfilename('fullpath');
thisDir  = fileparts(thisFile);
addpath(fullfile(thisDir, "separateSpeakers"));

%% PARAMS
fs = 16000; %16 KHz sampling freq
c = 340; %Speed of sound, m/s
roomDims = [4.9 4.9 4.9]; %LxBxH (m)

nfft = 512;
winlen = 512;
hop = 128;
win = hann(winlen, 'periodic');

epsv = 1e-12;

%% SIMULATION ENVIRONMENT
mic1 = [2.41 2.45 1.5];
mic2 = [2.49 2.45 1.5];
micPos = [mic1; mic2];
d_mic = norm(mic1(1:2) - mic2(1:2));

src_target = [2.45 3.45 1.5];
src_interf = [3.22 3.06 1.5];

[male, fs1] = audioread("./InputSignals/target_clean.wav");
[fem,  fs2] = audioread("./InputSignals/interference_clean.wav");

if fs1 ~= fs, male = resample(male, fs, fs1); end
if fs2 ~= fs, fem  = resample(fem,  fs, fs2); end

L = min(length(male), length(fem)); %Making sure the length of both signals is same
male = male(1:L);
fem  = fem(1:L);

ref_clean = male; %Reference signal
interf_clean = fem;

if cfg.useDereverb == 1
    ref_clean = dereverb_single_channel(ref_clean, fs, dereverbNet);
end

%% SIGNAL MODEL AND PROPAGATION
%mic_n(t) = h_a(t)*s(t) + h_b(t)*i(t) + n(t)

%Room Impulse Responses (RIRs)
if cfg.task == 1
    reflOrder = 0;
else
    reflOrder = 4;
end

h_t1 = acousticRoomResponse(roomDims, src_target, mic1, ...
    Algorithm="image-source", ImageSourceOrder=reflOrder, SampleRate=fs);
h_t2 = acousticRoomResponse(roomDims, src_target, mic2, ...
    Algorithm="image-source", ImageSourceOrder=reflOrder, SampleRate=fs);
h_i1 = acousticRoomResponse(roomDims, src_interf, mic1, ...
    Algorithm="image-source", ImageSourceOrder=reflOrder, SampleRate=fs);
h_i2 = acousticRoomResponse(roomDims, src_interf, mic2, ... 
    Algorithm="image-source", ImageSourceOrder=reflOrder, SampleRate=fs);

%Propagation
t1 = filter(h_t1,1,male); %Linear convolution
t2 = filter(h_t2,1,male);
i1 = filter(h_i1,1,fem);
i2 = filter(h_i2,1,fem);

scale = sqrt(mean(t1.^2) / mean(i1.^2)); %Ensuring target and interferer to be at same power (0 dB)
i1 = i1 * scale;
i2 = i2 * scale;

mix1 = awgn(t1 + i1, 5, 'measured'); %AWGN added (5 dB SNR)
mix2 = awgn(t2 + i2, 5, 'measured');

audiowrite("reverberant_mixture_mic1.wav", mix1, fs);

%Begin speech processing pipeline
tic %measures time required for computation

%% DEREVERBERATION
if cfg.useDereverb == 1
    mix1 = dereverb_single_channel(mix1, fs, dereverbNet);
    mix2 = dereverb_single_channel(mix2, fs, dereverbNet);
end

%% BEAMFORMING
params = struct( ...
    'fs',fs,'c',c, ...
    'nfft',nfft,'winlen',winlen,'hop',hop,'win',win, ...
    'micPos',micPos,'steerAzDeg',90, ...
    'delta_base',5e-3,'f_shoulder',200,'epsv',epsv);

[y_bf, W] = run_mvdr_fast(mix1, mix2, params);
y_bf_linear = y_bf; %Save snapshot of beamformed output for SI-SDR evaluation of beamformer

%% Conv-TasNet 
%Use Conv-TasNet model in separateSpeakers method with one-rest seperation
%source: https://in.mathworks.com/help/audio/ug/compare-speaker-separation-models.html
y_bf = separateSpeakers(y_bf, fs, NumSpeakers=1);

%% DE-NOISING
y_es = enhanceSpeech(y_bf, fs);
rms_ref = sqrt(mean(y_bf.^2) + epsv);
rms_1   = sqrt(mean(y_es.^2) + epsv);
y_bf    = 0.98 * y_es * (rms_ref / rms_1) / max(abs(y_es)+epsv);
y_norm = 0.98 * y_bf / max(abs(y_bf) + epsv);

%end speech processing pipeline
toc 
%Write audio files
audiowrite(artifact.enhancedWav, y_norm, fs); %write final output audio file
ref_norm = ref_clean(1:length(y_norm));
ref_norm = ref_norm / max(abs(ref_norm) + epsv);
audiowrite(artifact.refWav, ref_norm, fs); %write reference audio file

%% METRIC EVALUATION

%SI-SDR Improvement
ref_sisdr = male(1:length(y_bf_linear));
mix_sisdr = mix1(1:length(ref_sisdr));

SI_SDR_BF  = compute_si_sdr(y_bf_linear, ref_sisdr);
SI_SDR_mix = compute_si_sdr(mix_sisdr, ref_sisdr);
SI_SDR_i   = SI_SDR_BF - SI_SDR_mix;

%STOI
STOI_val = stoi(y_bf, ref_sisdr, fs);

%ViSQOL
visqol_val = visqol(y_bf, ref_sisdr, fs);

%PESQ
mos = compute_pesq(artifact.refWav, artifact.enhancedWav, fs);

%Print metrics
fprintf('\n============== RESULTS ==============\n');
fprintf('SI-SDR (BF stage) : %.2f dB\n', SI_SDR_BF);
fprintf('SI-SDR (mixture)  : %.2f dB\n', SI_SDR_mix);
fprintf('SI-SDR improvement: %.2f dB\n', SI_SDR_i);
fprintf('STOI              : %.3f\n', STOI_val);
fprintf('ViSQOL            : %.3f\n', visqol_val);
fprintf('PESQ              : %.3f\n', mos);
fprintf('=======================================\n');

%% .MAT EXPORT

% --- Required signals ---
target_signal       = male(:);                % clean male target
interference_signal = fem(:);                 % clean interferer
mixture_signal      = [mix1(:), mix2(:)];     % 2-channel mixture
processed_signal    = y_bf(:);                % final enhanced output

% --- RIR packaging ---
rir_data = struct( ...
    'target', struct('mic1', h_t1(:), 'mic2', h_t2(:)), ...
    'interference', struct('mic1', h_i1(:), 'mic2', h_i2(:)), ...
    'reflOrder', reflOrder, ...
    'roomDims', roomDims );

metrics = struct( ...
    'SI_SDR_i_dB', SI_SDR_i, ...
    'STOI', STOI_val, ...
    'ViSQOL', visqol_val, ...
    'PESQ', mos );

% --- Parameters / metadata ---
params = struct( ...
    'task', cfg.task, ...
    'fs', fs, ...
    'mic_positions', micPos, ...
    'array_spacing', d_mic, ...
    'target_position', src_target, ...
    'interference_position', src_interf, ...
    'target_azimuth_deg', 90, ...
    'SNR_dB', 5, ...
    'SIR_dB', 0 );

% --- Save ---
matName = sprintf('SPCup_%s.mat', artifact.taskTag );
save(matName, ...
    'target_signal', ...
    'interference_signal', ...
    'mixture_signal', ...
    'rir_data', ...
    'processed_signal', ...
    'metrics', ...
    'params', ...
    '-v7.3');

fprintf('\nSaved SPCup-compliant MAT file: %s\n', matName);

%% PLOTS
L = length(y_bf);
t = (0:L-1)/fs;

subplot(5,1,1);
plot(t, ref_clean);
title('Clean target speaker signal')
xlabel('Time (s)');
ylabel('Amplitude');

subplot(5,1,2);
plot(t, interf_clean);
title('Clean interferer signal')
xlabel('Time (s)');
ylabel('Amplitude');

subplot(5,1,3);
plot(t, mix1);
title('Mixture received at mic1')
xlabel('Time (s)');
ylabel('Amplitude');

subplot(5,1,4);
plot(t, mix2);
title('Mixture received at mic2')
xlabel('Time (s)');
ylabel('Amplitude');

subplot(5,1,5);
plot(t, y_bf);
title('Final processed output')
xlabel('Time (s)');
ylabel('Amplitude');


%% FUNCTIONS
function y = dereverb_single_channel(x, fs, dereverbNet)
%Specifications tuned according MathWorks article: https://www.mathworks.com/help/audio/ug/dereverberate-speech-using-deep-learning-networks.html
%STFT params
params.WindowLength  = 512;
params.Window        = hamming(params.WindowLength,"periodic");
params.OverlapLength = round(0.75*params.WindowLength);
params.FFTLength     = params.WindowLength;

x = single(x(:)); %convert data vector to 32-bit single-precision floating-point column vector

%STFT
X = stft(x, ...
    Window=params.Window, ...
    OverlapLength=params.OverlapLength, ...
    FFTLength=params.FFTLength, ...
    FrequencyRange="onesided");

Eps = realmin("single"); %To prevent indefinite log values

magLog = log(abs(X(1:end-1,:)) + Eps);
phase  = angle(X(1:end-1,:));

%Data preprocessing
%split log-magnitude spectrogram into fixed-length time chunks, normalize each chunk independently to [−1, 1], and reshape everything into a 4-D tensor suitable for a DereverbNet input

numFeat = 256;
numSeg  = 256;
T = size(magLog,2);
chunks = ceil(T/numSeg);

segments = mat2cell(magLog, numFeat, ...
    [numSeg*ones(1,chunks-1), T-(chunks-1)*numSeg]);

segments{chunks} = magLog(:,end-numSeg+1:end);

% Normalize
minVals = cellfun(@(x)min(x,[],"all"),segments,'UniformOutput',false);
maxVals = cellfun(@(x)max(x,[],"all"),segments,'UniformOutput',false);

featNorm = cellfun(@(x,mn,mx) ...
    2*(x-mn)/(mx-mn)-1, ...
    segments,minVals,maxVals,'UniformOutput',false);

featNorm = reshape(cell2mat(featNorm),numFeat,numSeg,1,chunks); %featNorm ∈ ℝ^(freq × time × channels × batch) = 256 × 256 × 1 × chunks

%predict
Yhat4D = predict(dereverbNet,featNorm);

%Data post-processing
Yhat = squeeze(mat2cell(Yhat4D,numFeat,numSeg,1,ones(1,chunks)))';
Yhat = cellfun(@(x,mn,mx) ...
    (x+1)*(mx-mn)/2 + mn, ...
    Yhat,minVals,maxVals,'UniformOutput',false);

Yhat = cellfun(@exp,Yhat,'UniformOutput',false);

% Reconstruct full spectrogram
Yfull = cat(2,Yhat{1:chunks-1});
Yfull(:,T-numSeg+1:T) = Yhat{chunks};

% Restore missing bin
Yfull = cat(1,Yfull,zeros(1,T,'single'));
phase = cat(1,phase,zeros(1,T,'single'));

% ISTFT
Xrec = Yfull .* exp(1j*phase);
y = istft(Xrec, ...
    Window=params.Window, ...
    OverlapLength=params.OverlapLength, ...
    FFTLength=params.FFTLength, ...
    ConjugateSymmetric=true, ...
    FrequencyRange="onesided");

% Length + normalize
y = y(1:length(x));
y = y / max(abs(y)+eps);
end

function [y_out,W] = run_mvdr_fast(x1,x2,p)
x1=x1(:); 
x2=x2(:); 
N=min(length(x1),length(x2));
numFrames=ceil((N-p.winlen)/p.hop)+1;
halfBins=p.nfft/2+1; %one-sided FFT, real signal
Xf=zeros(halfBins,2,numFrames);
ptr=1;

%Manual STFT
for m = 1:numFrames
    x1f = fft(x1(ptr:ptr+p.winlen-1).*p.win, p.nfft);
    x2f = fft(x2(ptr:ptr+p.winlen-1).*p.win, p.nfft);

    Xf(:,:,m) = [x1f(1:halfBins), x2f(1:halfBins)];
    ptr = ptr + p.hop;
end

%Contruct steering vector
freqs=(0:halfBins-1)'*p.fs/p.nfft;
az=deg2rad(p.steerAzDeg); %azimuth angle
d=(p.micPos(2,1:2)-p.micPos(1,1:2)).';
u=[cos(az);sin(az)];
td=-dot(d,u)/p.c; %time delay

A=[ones(halfBins,1),exp(-1j*2*pi*freqs*td)]; %steering vector

%Covariance estimation for each frequency
R = zeros(2,2,halfBins);
for k=1:halfBins
    Xk=squeeze(Xf(k,:,:));
    Rk=(Xk*Xk')/numFrames;
    R(:,:,k)=Rk + eye(2)*trace(Rk)*1e-3; %Diagonal loading to improve MVDR performance
end

%Compute MVDR weights according solution of constraint optimization
W=zeros(2,halfBins);
for k=1:halfBins
    ak=A(k,:)';
    W(:,k)=(R(:,:,k)\ak)/(ak'*(R(:,:,k)\ak)+p.epsv);
end

%Apply beamformer in frequency domain
Y=zeros(halfBins,numFrames);
for m=1:numFrames
    for k=1:halfBins
        Y(k,m)=W(:,k)'*squeeze(Xf(k,:,m)).';
    end
end

%Manual ISTFT
sigLen = (numFrames-1)*p.hop + p.winlen;
y = zeros(sigLen,1);
ptr = 1;
for m = 1:numFrames
    Yfull = [Y(:,m); conj(Y(end-1:-1:2,m))];
    yt = real(ifft(Yfull, p.nfft)) .* p.win;
    y(ptr:ptr+p.winlen-1) = y(ptr:ptr+p.winlen-1) + yt;
    ptr = ptr + p.hop;
end

y_out = y(1:N);
end

%Metrics
%SI-SDR
function si_sdr = compute_si_sdr(s_hat, s_ref)
epsv = 1e-12;
s_hat = s_hat(:); s_ref = s_ref(:);
L = min(length(s_hat),length(s_ref));
s_hat = s_hat(1:L); s_ref = s_ref(1:L);
alpha = (s_hat'*s_ref)/(s_ref'*s_ref+epsv);
e = s_hat - alpha*s_ref;
si_sdr = 10*log10((alpha^2*sum(s_ref.^2)+epsv)/(sum(e.^2)+epsv));
end

%PESQ Wrapper
function mos = compute_pesq(ref_wav, deg_wav, fs)
assert(exist(ref_wav,'file')==2, 'Reference wav not found');
assert(exist(deg_wav,'file')==2, 'Degraded wav not found');

if fs == 8000
    mode = '+8000';
elseif fs == 16000
    mode = '+16000';
else
    error('PESQ supports only 8 kHz or 16 kHz');
end

pesq_bin = fullfile(pwd,'pesq.exe');
cmd = sprintf('"%s" %s "%s" "%s"', ...
    pesq_bin, mode, ref_wav, deg_wav);

[status, out] = system(cmd);
if status ~= 0
    error('PESQ failed:\n%s', out);
end
% Robust MOS parsing
tok = regexp(out, 'PESQ_MOS\s*=\s*([0-9.]+)', 'tokens');
if isempty(tok)
    error('Could not parse PESQ output:\n%s', out);
end

mos = str2double(tok{1}{1});
end