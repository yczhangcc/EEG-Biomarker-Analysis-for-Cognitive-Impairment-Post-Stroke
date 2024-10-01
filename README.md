# 🧠 EEG Biomarker Analysis for Cognitive Impairment Post-Stroke

This project focuses on analyzing EEG biomarkers to assess different levels of cognitive impairment in stroke patients. Specifically, it uses Power Spectral Density (PSD) and functional connectivity to explore neural activity.

## 📋 Table of Contents
- [Overview](#overview)
- [📁 Folder Structure](#folder-structure)
- [📊 Data](#data)
- [🔧 Dependencies](#dependencies)
- [🚀 Usage](#usage)
- [🔗 References](#references)

## 🔍 Overview

This project builds upon research exploring EEG biomarker analysis in stroke patients with varying degrees of cognitive impairment. The key methods include:
- **📈 Power Spectral Density (PSD)**: Analyzes brain wave frequencies (delta, theta, alpha, beta, gamma) to identify cognitive states.
- **🌐 Functional Connectivity**: Examines connectivity between brain regions to understand cognitive decline.

For more details, refer to:
- [📝 EEG Biomarkers in Stroke Cognitive Impairment](https://www.frontiersin.org/articles/10.3389/fneur.2024.1358167/full)
- [📖 Advanced Cognitive and Neuroimaging Techniques](https://www.frontiersin.org/articles/10.3389/fnins.2023.1269359/full)

## 📁 Folder Structure

```
.
├── avg_final_picture/
├── datasetqw/
├── microstate/
├── picture/
├── plotset_process_note_vision/
├── plotsetavg_final_vision/
├── Scripts/
├── singel_mne_result/
├── sperate_power/
├── avg_plot_conc_metrix_circle.py
├── plotsingle.py
├── pyenv/
├── .gitignore
└── README.md
```

- **🖼 avg_final_picture/**: Contains final visualizations of EEG biomarkers.
- **📊 datasetqw/**: Processed EEG datasets for cognitive impairment levels.
- **🧩 microstate/**: Microstate analysis results.
- **🖼 picture/**: Images generated during analysis.
- **📝 plotset_process_note_vision/**: Notes and intermediate visualizations related to the analysis.
- **📈 plotsetavg_final_vision/**: Final averaged EEG visualizations.
- **⚙️ Scripts/**: Python scripts for EEG preprocessing, PSD analysis, and functional connectivity visualization.
- **📂 singel_mne_result/**: MNE-Python EEG analysis results.
- **📉 sperate_power/**: Power spectral analysis results.
- **📊 avg_plot_conc_metrix_circle.py**: Script for plotting average connectivity circles.
- **📊 plotsingle.py**: Script for plotting individual EEG signals.
- **💻 pyenv/**: Python virtual environment configuration files.
- **🚫 .gitignore**: Specifies files to be ignored by version control.

## 📊 Data

The data used in this project includes EEG recordings from stroke patients at varying levels of cognitive impairment. The key analysis focuses on PSD and functional connectivity in different brain regions.

- **👨‍⚕️ Participants**: 32 total participants (8 healthy controls and 24 stroke patients with mild, moderate, and severe cognitive impairment).
- **💻 EEG Recording**: 44-channel EEG data collected at 2 kHz, analyzed using MNE-Python.

## 🔧 Dependencies

Ensure you have the following dependencies installed:

- Python 3.8+
- MNE-Python
- EEGLAB
- numpy
- matplotlib
- scipy

You can install the required Python packages using:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Activate the Python virtual environment**:
   ```bash
   source pyenv/bin/activate
   ```

3. **Run the scripts**:
   - To generate PSD plots:
     ```bash
     python avg_plot_conc_metrix_circle.py
     ```
   - To visualize individual EEG data:
     ```bash
     python plotsingle.py
     ```

4. **Review the output** in the `avg_final_picture/` and `picture/` folders.

## 🔗 References

1. Xu, M., Zhang, Y., Zhang, Y., Liu, X., & Qing, K. (2024). **EEG biomarkers analysis in different cognitive impairment after stroke: an exploration study**. *Frontiers in Neurology*, 15:1358167. [Link to article](https://www.frontiersin.org/articles/10.3389/fneur.2024.1358167/full)
2. Zhang, Y., Zhang, Y., Jiang, Z., Xu, M., & Qing, K. (2023). **The effect of EEG and fNIRS in the digital assessment and therapy of Alzheimer’s disease**. *Frontiers in Neuroscience*, 17:1269359. [Link to article](https://www.frontiersin.org/articles/10.3389/fnins.2023.1269359/full)

---

Let me know if you'd like to add or modify anything further!
