# Cognitive-Stress-Analysis
This project provides a framework for analyzing speech patterns to detect potential cognitive stress. By combining acoustic feature extraction and linguistic analysis, the system evaluates audio input to generate quantitative metrics and comparative assessments of cognitive load.

## Objective
To develop an automated system that:
- Converts speech to text
- Extracts acoustic and linguistic features
- Applies unsupervised machine learning
- Generates comparative analysis reports
- Identifies patterns indicative of cognitive stress

## What is Cognitive Stress?
Cognitive stress refers to mental strain occurring when:
- Working memory demands exceed capacity
- Performing complex information processing
- Experiencing emotional or psychological pressure
- Managing competing cognitive tasks

Manifested in speech through:
- Increased hesitations (um/uh)
- Irregular pitch patterns
- Frequent pauses
- Reduced speech rate
- Simplified sentence structures

## Features
- **Audio Preprocessing**: Converts audio files to text using Google Speech Recognition
- **Multi-Dimensional Analysis**:
  - Acoustic: Speech rate, pitch characteristics, pause patterns
  - Linguistic: Hesitation markers, sentence complexity, word metrics
- **Machine Learning Integration**:
  - K-Means clustering (2 clusters)
  - Isolation Forest anomaly detection
  - Cosine similarity scoring
- **Automated Reporting**: Generates CSV files with detailed analysis metrics
- **Comparative Analysis**: Provides cluster classification and anomaly scores

## Technologies Used
- **Core Libraries**:
  - `librosa` (audio analysis)
  - `SpeechRecognition` (STT conversion)
  - `nltk` (linguistic processing)
  - `scikit-learn` (machine learning)
- **Processing**:
  - `pandas` (data handling)
  - `numpy` (numerical operations)
- **Machine Learning Models**:
  - K-Means clustering
  - Isolation Forest
  - Cosine Similarity scoring





## Implementation Overview
```python
# Core Workflow
audio_files = ['sample.wav']
feature_df = process_audio_samples(audio_files)
results_df = apply_unsupervised_learning(feature_df)
results_df.to_csv('cognitive_analysis_results.csv')
```

## Output Metrics (CSV Columns)
1. Acoustic Features:
   - Speech Rate (words/min)
   - Mean Pitch (Hz)
   - Pitch Standard Deviation
   - Average Pause Duration
   - Pause Frequency

2. Linguistic Features:
   - Hesitation Count
   - Word Count
   - Average Sentence Length
   - Hesitation Ratio

3. Machine Learning Outputs:
   - Cluster Assignment (0/1)
   - Anomaly Score (-1=outlier, 1=normal)
   - Similarity Score (0-1)

## Usage
1. Install requirements:
```bash
pip install -r requirements.txt
```
2. Place audio files in `/input` directory
3. Run main analysis:
```python
python cognitive_stress_analyzer.py
```
4. Find results in:
- `cognitive_analysis_results.csv`
- Console output of cluster classifications


## Applications
1. **Mental Health Screening**: Early detection of stress/anxiety disorders
2. **Workplace Monitoring**: Employee well-being assessment
3. **Education**: Student performance evaluation during exams
4. **Healthcare**: Post-stroke cognitive assessment
5. **Call Centers**: Agent stress level monitoring
6. **Public Speaking**: Presentation skills analysis

## Conclusion
This system provides a non-invasive method for cognitive stress assessment through speech pattern analysis. By combining traditional NLP techniques with modern machine learning approaches, it offers actionable insights for various professional domains. The unsupervised learning approach makes it particularly suitable for scenarios where labeled training data is scarce.

## Future Enhancements
- Real-time analysis capability
- Web interface for audio submission
- Advanced deep learning models
- Multi-language support
- Personalized baseline creation

## Ethical Considerations
- Ensure proper user consent for audio recording
- Maintain data privacy and security
- Use results as supplementary information only
- Provide clear disclaimers about medical limitations

---

**Note**: This system is designed for assistive purposes and should not be used as a standalone diagnostic tool. Professional evaluation is recommended for clinical applications.


