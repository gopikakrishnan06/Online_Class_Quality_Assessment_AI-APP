# Online Class Quality Assessment AI

## Introduction
An AI-assisted prototype for assessing online classroom recordings using audio, visual, and transcript-based features. It extracts measurable indicators and combines selected assessment scores using a weighted scoring system to generate an assessment report.

## Key Features

### 1. Audio Analysis
- RMS energy and peak amplitude
- Silence percentage and clipping indicators
- Audio duration and signal characteristics

### 2. Visual Analysis
- Video-frame sampling
- Brightness, contrast, and sharpness
- Recording-quality indicators

### 3. Speech Transcription
- Speech-to-text using Faster-Whisper
- Transcript generation for further analysis

### 4. Transcript Analysis
- Lesson introduction, explanation, and conclusion
- Questioning, encouragement, and recap indicators
- Word count and language features

### 5. Teaching Style Indicators
- Inclusiveness and encouragement
- Student engagement and interaction
- Communication and explanation

### 6. Scoring and Classification
- Category-wise assessment scores
- Weighted scoring and overall score
- Provisional ratings: Excellent, Good, Average, and Poor

### 7. Interactive Dashboard
- Assessment summaries and visualisations
- Audio, visual, and transcript results
- Downloadable reports in supported formats

## Requirements
- Python 3.10+
- Dependencies listed in `requirements.txt`
- A supported classroom recording

## Note
This is an experimental prototype. Assessment indicators and scoring thresholds require further validation. Results should not be used as the sole basis for formal teacher evaluation.
