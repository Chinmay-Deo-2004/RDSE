<img width="1917" height="588" alt="image" src="https://github.com/user-attachments/assets/8ee6c4b8-4562-418b-82d6-a8c8ead246d2" />

# RDSE: Real-Time Directional Speech Enhancement

This repository maintains and documents the open-source MATLAB implementation of **RDSE (Real-Time Directional Speech Enhancement Framework for Dual-Mic Commodity Smartphones)**.

The repository includes MATLAB scripts for offline audio processing and live recording, pretrained models, sample audio recordings, and evaluation utilities to facilitate reproducibility and experimentation.

## Prerequisites and Setup

### 1. MATLAB Environment

Ensure MATLAB is installed along with the required toolboxes and dependencies for the scripts.

### 2. Download `dereverbNet.mat`

Due to GitHub file-size limitations, `dereverbNet.mat` needs to be downloaded separately.

Run the following code in MATLAB to download and load the pretrained dereverberation network:

```matlab
downloadFolder = matlab.internal.examples.downloadSupportFile( ...
    "audio/examples", "dereverbunet.zip");

dataFolder = tempdir;
netFolder = fullfile(dataFolder, "dereverbunet");

unzip(downloadFolder, netFolder);

load(fullfile(netFolder, "dereverbNet.mat"));
```

This downloads the supporting archive, extracts it to a temporary directory, and loads the network into the MATLAB workspace.

**Note:** The code above loads the network from MATLAB's temporary directory. If the scripts expect `dereverbNet.mat` in the repository root, copy the downloaded file there or update the corresponding model path.

### 3. Other Pretrained Models

The other pretrained models used in this study are included in the repository for convenience and reproducibility:

- `seprateSpeakers/` — Contains pretrained speaker separation models.
- `metric-gan-okd.mat` — Contains the pretrained MetricGAN model.

These models are MATLAB-owned assets included to facilitate the replication of this study. Please consult the applicable MATLAB documentation and licensing terms for their usage and redistribution conditions.

### 4. PESQ Evaluation Utility

The repository includes `pesq.exe`, created using the official PESQ documentation, to simplify objective speech-quality evaluation and facilitate reproducibility.

Ensure that the executable is accessible to the relevant MATLAB scripts and that its usage complies with the applicable PESQ licensing terms.

## Running RDSE

### 1. Offline Audio Processing

Run `RDSE.m` in MATLAB to explore the framework using a fully post-processing-based pipeline.

Example audio paths are configured in the `inputSignals/` directory. The included sample recordings are taken from the **LibriSpeech** and **MUSAN** datasets.
These samples provide a convenient starting point for exploring the framework without requiring a live recording session.

### 2. Live Recording Demonstration

One of the most challenging and exciting parts of this project was implementing the live recording pipeline.

To see RDSE in action, run [`run_live_recording_pipeline.m`](https://github.com/Chinmay-Deo-2004/RDSE/blob/main/run_live_recording_pipeline.m).

#### Requirements

The live demonstration requires two devices or MATLAB sessions:

1. **Smartphone:** Use MATLAB Mobile to record audio live.
2. **PC/Laptop:** Run `run_live_recording_pipeline.m` to receive and process the audio.

**Important:** MATLAB Mobile does not currently support the simultaneous streaming and processing workflow required by this implementation. Therefore, the recording and processing operations must be distributed across the smartphone and the PC/laptop.

Refer to `run_live_recording_pipeline.m` for the relevant configuration and execution sequence.

## Repository Structure

Your local repo should look like this:

```text
RDSE/
├── inputSignals/
├── seprateSpeakers/
├── RDSE.m
├── run_live_recording_pipeline.m
├── metric-gan-okd.mat
├── pesq.exe
├── dereverbNet.mat
└── README.md
```

## Citation

If you use RDSE in your research, please cite the associated arXiv preprint:

```bibtex
@misc{deo2026rdserealtimedirectionalspeech,
    title         = {RDSE: Real-Time Directional Speech Enhancement Framework for Dual-Mic Commodity Smartphones},
    author        = {Chinmay Deo and Danish Mahajan and Soudeep Ghoshal and Himanshu Buckchash},
    year          = {2026},
    eprint        = {2610.12476},
    archivePrefix = {arXiv},
    primaryClass  = {cs.SD},
    url           = {https://arxiv.org/abs/2610.12476}
}
```

**Preprint:** [arXiv:2610.12476](https://arxiv.org/abs/2610.12476)

## Acknowledgements

RDSE builds upon pretrained models and supporting resources provided by MATLAB, as well as publicly available audio datasets. Please consult the respective sources for applicable licenses, attribution requirements, and usage conditions.
