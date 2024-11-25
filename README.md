# Laboratorium AI Forced Alignment

![Python](https://img.shields.io/badge/Python-3.10.13-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Poetry](https://img.shields.io/badge/Build-Poetry-blue.svg)

**Laboratorium AI Forced Alignment** is a Python package for forced alignment of transcriptions with audio recordings. It processes `.wav` audio files along with existing transcription data in `.json` files and generates precise timestamps for sentences and words. The package uses modern speech models and libraries like `torchaudio` and `nltk` for efficient and accurate results.

## Table of Contents

- [Laboratorium AI Forced Alignment](#laboratorium-ai-forced-alignment)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Key Features](#key-features)
  - [Installation and Build](#installation-and-build)
  - [Usage](#usage)
    - [CLI Usage with File Descriptors](#cli-usage-with-file-descriptors)
      - [1. Start Module and Load Model](#1-start-module-and-load-model)
      - [2. Wait for "ready" Signal](#2-wait-for-ready-signal)
      - [3. Process Files](#3-process-files)
    - [Example with Shell Script](#example-with-shell-script)
  - [License](#license)

## Overview

**Laboratorium AI Forced Alignment** provides a powerful solution for synchronizing transcriptions with their corresponding audio recordings. This is particularly useful for applications like subtitling, speech research, and automated content analysis.

### Key Features

- **Forced Alignment:** Precise mapping of transcription sentences and words to corresponding audio timestamps.
- **Model Variety:** Utilization of advanced speech models for accurate results.
- **Text Normalization:** Includes conversion of numbers to words and umlaut adjustments.
- **Flexible Configuration:** Adjustment of device (`DEVICE`) and computation type (`COMPUTE_TYPE`) for optimal performance and accuracy.

## Installation and Build

This package is managed with [Poetry](https://python-poetry.org/). Follow these steps to install and build the package:

1. **Clone Repository:**

   ```bash
   git clone https://github.com/uzl-cosy/cosy-ai-forced-alignment.git
   cd laboratorium-ai-forced-alignment
   ```

2. **Install Dependencies:**

   ```bash
   poetry install
   ```

3. **Activate Virtual Environment:**

   ```bash
   poetry shell
   ```

4. **Build Package:**

   ```bash
   poetry build
   ```

   This command creates the distributable files in the `dist/` directory.

## Usage

The package runs as a continuously running module via command line (CLI). It enables forced alignment of transcription data and audio recordings using file descriptors. Communication occurs through a pipe, where the module sends "ready" once the model is loaded and ready for processing.

### CLI Usage with File Descriptors

#### 1. Start Module and Load Model

Start the Forced Alignment module via CLI. The module loads the model and signals through the file descriptor when it's ready.

```bash
python -m laboratorium_ai_forced_alignment -f <FD>
```

**Parameters:**

- `-f`, `--fd`: File descriptor for pipe communication.

**Example:**

```bash
python -m laboratorium_ai_forced_alignment -f 3
```

#### 2. Wait for "ready" Signal

After the module starts, it loads the Forced Alignment model. Once loaded, the module sends a "ready" signal through the specified file descriptor.

#### 3. Process Files

Pass the paths to input (`.json` for transcription and `.wav` for audio) and output (`.json` for alignment) files through the pipe. The module processes the files and sends a "done" signal once alignment is complete.

**Example:**

```bash
echo "path/to/input_transcript.json,path/to/input_audio.wav,path/to/output_alignment.json" >&3
```

- **Description:**
  - The `echo` command sends the input and output file paths through file descriptor `3`.
  - The module receives the paths, performs forced alignment, and saves the result in the `.json` file.
  - Upon completion, the module sends a "done" signal through the file descriptor.

**Complete Flow:**

1. **Start the Forced Alignment module:**

   ```bash
   python -m laboratorium_ai_forced_alignment -f 3
   ```

2. **Send file paths for alignment:**

   ```bash
   echo "path/to/input_transcript.json,path/to/input_audio.wav,path/to/output_alignment.json" >&3
   ```

3. **Repeat step 2 for additional files:**

   ```bash
   echo "path/to/another_input_transcript.json,path/to/another_input_audio.wav,path/to/another_output_alignment.json" >&3
   ```

### Example with Shell Script

Here's an example of how to use the Forced Alignment package in a shell script:

```bash
#!/bin/bash

# Open a file descriptor (e.g., 3) for pipe communication

exec 3<>/dev/null

# Start the Forced Alignment module in background and connect the file descriptor

python -m laboratorium_ai_forced_alignment -f 3 &

# Store PID of Forced Alignment module to terminate it later

FA_PID=$!

# Wait for "ready" signal

read -u 3 signal
if [ "$signal" = "ready" ]; then
      echo "Model is ready for processing."

      # Send input and output paths
      echo "path/to/input_transcript.json,path/to/input_audio.wav,path/to/output_alignment.json" >&3

      # Wait for "done" signal
      read -u 3 signal_done
      if [ "$signal_done" = "done" ]; then
            echo "Alignment completed."
      fi

      # Additional alignments can be added here
      echo "path/to/another_input_transcript.json,path/to/another_input_audio.wav,path/to/another_output_alignment.json" >&3

      # Wait again for "done" signal
      read -u 3 signal_done
      if [ "$signal_done" = "done" ]; then
            echo "Additional alignment completed."
      fi

fi

# Close the file descriptor

exec 3>&-
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
