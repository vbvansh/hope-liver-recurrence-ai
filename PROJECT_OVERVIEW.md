# HOPE — AI for Earlier Detection of Liver Cancer Recurrence

## Project Overview

**Project Name:** HOPE — Earlier Detection of Liver Cancer Recurrence
**Repository:** `hope-liver-recurrence-ai`
**GitHub:** https://github.com/vbvansh/hope-liver-recurrence-ai
**Project Page:** https://vivek231.github.io/HOPE/
**Institution:** Barts Cancer Institute, Queen Mary University of London
**Project Type:** 12-month proof-of-concept research project
**Research Area:** Medical AI, Liver Cancer, HCC, Longitudinal Medical Imaging, Multimodal AI

---

# 1. What is HOPE?

HOPE is a research project focused on **earlier detection of liver cancer recurrence**, particularly recurrence of **hepatocellular carcinoma (HCC)** after treatment.

The central idea is to combine two different types of information:

1. **Longitudinal contrast-enhanced CT scans**
2. **Longitudinal blood-test measurements, especially AFP**

Instead of looking at only one CT scan at a time, the project studies how the patient's liver changes across multiple follow-up scans.

The project also studies whether changes in AFP over time can provide additional information when combined with imaging.

The long-term goal is to determine whether artificial intelligence can identify signs of recurrence **earlier than routine surveillance**, especially when the recurrent lesion is still extremely small.

---

# 2. Why is this problem important?

Patients who have previously been treated for liver cancer require regular surveillance because the cancer can return.

A recurrent tumor may initially be:

- extremely small
- difficult to see clearly
- difficult to distinguish from normal tissue
- difficult to distinguish from treatment-related changes
- below the size at which routine interpretation becomes reliable

For example, a recurrence may initially be only a few millimeters in size.

By the time it becomes clearly visible, valuable time may already have passed.

Therefore, one important research question is:

> Can AI identify subtle changes earlier by looking at the patient's imaging history and biomarker history together?

---

# 3. What is HCC?

**HCC = Hepatocellular Carcinoma.**

It is the most common type of primary liver cancer.

The HOPE project is particularly concerned with identifying recurrence of HCC after treatment.

---

# 4. What is AFP?

**AFP = Alpha-Fetoprotein.**

AFP is a protein that can be measured using a blood test.

AFP is normally produced at high levels during fetal development and is usually present at much lower levels in healthy adults.

AFP can become elevated in some liver diseases and in some patients with HCC.

However:

> AFP is NOT a perfect cancer detector.

A patient can have HCC with a normal AFP level, and AFP can also increase for reasons other than cancer.

Therefore, AFP is most useful when interpreted together with other information such as imaging and clinical history.

---

# 5. Why is AFP useful for this project?

The important concept is not only the AFP value at one particular time.

The project is interested in the **AFP trajectory**.

For example:

```text
January   → AFP = 8
April     → AFP = 12
July      → AFP = 25
October   → AFP = 60
```

The important observation is that the value is changing over time.

The model can potentially learn that this trajectory may contain information about disease progression when combined with imaging.

Thus, HOPE is interested in:

```text
AFP at time t1
AFP at time t2
AFP at time t3
AFP at time t4
```

rather than only:

```text
AFP at one time point
```

---

# 6. What does "longitudinal CT" mean?

Longitudinal data means information collected from the **same patient at multiple points in time**.

For example:

```text
Patient
   │
   ├── CT Scan 1 — January
   ├── CT Scan 2 — April
   ├── CT Scan 3 — July
   └── CT Scan 4 — October
```

The AI therefore has access to the patient's imaging history.

This allows the model to study not only:

> "What does the liver look like?"

but also:

> "How has the liver changed?"

---

# 7. Independent CT analysis vs longitudinal CT analysis

## Independent analysis

A conventional image model might receive:

```text
CT January → Prediction
```

Then separately:

```text
CT April → Prediction
```

Then:

```text
CT July → Prediction
```

Each scan is treated largely as an individual input.

The model does not necessarily reason explicitly about the sequence of changes.

## Longitudinal analysis

A longitudinal model instead receives:

```text
CT January
      ↓
CT April
      ↓
CT July
      ↓
CT October
```

and attempts to understand the changes between the scans.

For example:

```text
January → no obvious lesion

April   → tiny suspicious region

July    → region becomes slightly larger / different

October → region becomes more clearly suspicious
```

The model can therefore learn that a particular region has **evolved over time**.

---

# 8. Simple analogy

A useful way to understand longitudinal imaging is to compare it with a video.

### Independent approach

The model sees:

```text
Frame 1
Frame 2
Frame 3
Frame 4
```

and analyzes them separately.

### Longitudinal approach

The model sees:

```text
Frame 1 → Frame 2 → Frame 3 → Frame 4
```

and tries to understand the progression.

Medical CT scans are not literally video frames, but the concept is similar:

> Multiple scans of the same patient provide information about how disease-related changes evolve over time.

---

# 9. Why small lesions are important

One of the important motivations of HOPE is the detection of **very small recurrent lesions**.

A lesion may initially be extremely difficult to identify.

For example:

```text
Time 1 → 3 mm
Time 2 → 4 mm
Time 3 → 6 mm
Time 4 → 9 mm
```

The first scan may not provide enough evidence by itself.

However, when the entire sequence is considered, the model may be able to recognize:

> "There is a suspicious region that is consistently changing."

This temporal information may provide additional evidence that is not available from a single scan alone.

---

# 10. CT + AFP: multimodal information

HOPE combines information from two different sources.

## Imaging information

CT provides visual and anatomical information:

- liver structure
- lesion appearance
- lesion location
- enhancement characteristics
- changes in anatomy over time

## Biomarker information

AFP provides a biological signal obtained from blood.

Therefore:

```text
                 Patient
                    │
       ┌────────────┴────────────┐
       │                         │
   CT sequence              AFP sequence
       │                         │
CT1 → CT2 → CT3 → CT4      AFP1 → AFP2 → AFP3 → AFP4
       │                         │
       └────────────┬────────────┘
                    ↓
             Multimodal AI
                    ↓
       Possible recurrence signal
```

This is a **multimodal** approach because different types of patient information are combined.

---

# 11. Why temporal alignment matters

The CT and AFP measurements need to be interpreted in relation to time.

For example:

```text
January
CT Scan
AFP measurement

April
CT Scan
AFP measurement

July
CT Scan
AFP measurement
```

This makes it possible to study relationships such as:

> Did AFP start increasing around the same period that a suspicious imaging change appeared?

Or:

> Did a change in AFP precede a visible imaging change?

Temporal alignment is therefore an important part of the research problem.

---

# 12. Main research question

A central research question for this project is:

> Can longitudinal AI analysis of contrast-enhanced CT scans combined with AFP trajectories improve the early detection of HCC recurrence compared with analyzing individual scans independently?

---

# 13. Main research objective

The primary objective is to investigate whether AI can improve sensitivity for detecting **very small recurrent lesions**, particularly lesions that may initially be difficult to identify using routine surveillance.

---

# 14. Lead-time

One important concept in this project is **lead-time**.

Lead-time asks:

> How much earlier can the AI identify a useful signal compared with routine detection?

For example:

```text
Routine detection:
October 2027

AI detection:
April 2027
```

Potential lead-time:

```text
Approximately 6 months
```

However, demonstrating true clinical benefit is more complicated than simply showing an earlier prediction.

The research must consider issues such as:

- false positives
- false alarms
- lead-time bias
- differences in follow-up timing
- clinical usefulness
- whether the earlier signal genuinely corresponds to recurrence

---

# 15. Clinical interpretability

A clinically useful system should ideally provide more than:

```text
Recurrence probability = 87%
```

It should also provide information that can help clinicians understand the prediction.

For example:

```text
Suspicious region:
Liver segment / approximate location

Confidence:
High

Reason:
Progressive imaging change across follow-up scans

Supporting information:
AFP trajectory increased over the same period
```

Visual localization or outlining of suspicious areas can therefore be important.

---

# 16. Potential model components

The project may investigate components such as:

- CNN-based image encoders
- Vision Transformers
- 3D medical imaging models
- temporal transformers
- recurrent architectures
- temporal attention
- cross-time attention
- cross-modal attention
- multimodal fusion
- biomarker encoders
- image registration
- lesion detection
- segmentation
- change detection
- interpretable prediction mechanisms

These are possibilities rather than fixed commitments.

The architecture should ultimately be driven by the research question, available data, computational resources, and evidence from the literature.

---

# 17. Research areas to investigate

The literature review should cover several related areas.

## A. HCC recurrence prediction

Research on predicting recurrence after treatment.

## B. HCC recurrence detection

Research on identifying recurrence during follow-up.

## C. Longitudinal medical imaging

Methods that analyze multiple scans acquired over time.

## D. CT-based liver lesion detection

AI for identifying liver lesions in CT.

## E. Small-lesion detection

Methods specifically designed for small or sub-centimeter lesions.

## F. AFP-based prediction

Methods using AFP and other blood biomarkers.

## G. Multimodal AI

Combining imaging, laboratory data and clinical information.

## H. Temporal modeling

Methods for learning disease progression.

## I. Medical imaging foundation models

Recent large pretrained models for CT/MRI and medical imaging.

## J. Clinical interpretability

Methods for explaining where and why the model predicts recurrence.

---

# 18. Important research questions

The project should continuously investigate questions such as:

1. Does longitudinal imaging improve detection compared with single-scan analysis?
2. Does AFP trajectory provide additional predictive information?
3. How should CT and AFP information be fused?
4. How should irregular follow-up intervals be handled?
5. How should missing AFP measurements be handled?
6. How should scans from different scanners or institutions be handled?
7. Can the model detect lesions smaller than 10 mm?
8. Can the model detect lesions smaller than 5 mm?
9. Can the system provide useful localization?
10. How much lead-time can realistically be obtained?
11. How many false positives will the model generate?
12. Does the model generalize to patients and institutions not seen during training?

---

# 19. Literature search strategy

The literature review should not only search for papers that directly mention "HOPE" or "HCC recurrence."

It should investigate the broader technical landscape.

Important search categories include:

```text
HCC recurrence
Liver cancer recurrence
Longitudinal CT
Longitudinal MRI
Temporal medical imaging
Temporal transformers
Medical image change detection
CT + AFP
Imaging + biomarkers
Small liver lesion detection
Sub-centimeter HCC
Multimodal medical AI
Medical imaging foundation models
Early cancer detection
Lead-time analysis
Clinical AI deployment
```

Useful research platforms currently being used include:

- Consensus: https://consensus.app/
- AlphaXiv Assistant: https://www.alphaxiv.org/assistant

Papers should be tracked in the project's literature matrix and summarized consistently.

---

# 20. Project roadmap

The project is planned as a 12-month proof-of-concept effort.

A high-level roadmap is:

## Months 1–3

- data preparation
- required approvals
- data cleaning
- annotation
- cohort understanding
- literature review
- baseline definition

## Months 1–8

- model development
- longitudinal imaging modeling
- biomarker modeling
- multimodal fusion
- experiments

## Months 6–10

- validation
- error analysis
- small-lesion analysis
- radiologist feedback
- interpretability analysis

## Months 11–12

- final evaluation
- reporting
- proof-of-concept results
- publication preparation
- planning for larger multicentre research

---

# 21. Project page

Official project page:

**https://vivek231.github.io/HOPE/**

The project page should be treated as an important reference for the overall project objectives, team, clinical motivation, and roadmap.

---

# 22. Core idea in one sentence

> **HOPE investigates whether AI can detect HCC recurrence earlier by learning how a patient's liver changes across sequential contrast-enhanced CT scans and combining those imaging changes with longitudinal AFP blood-test information.**

---

# 23. Useful external resources

**HOPE Project Page**
https://vivek231.github.io/HOPE/

**GitHub Repository**
https://github.com/vbvansh/hope-liver-recurrence-ai

**Consensus**
https://consensus.app/

**AlphaXiv Assistant**
https://www.alphaxiv.org/assistant

---
