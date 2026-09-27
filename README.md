# AeroVista — UAV Flight Intelligence & Business Intelligence

AeroVista is a student-developed **Data Mining and Business Intelligence project** for analysing UAV/drone flight behaviour.

The project uses a reference UAV trajectory dataset to identify common flight behaviour patterns and detect observations that differ from learned reference patterns. A Streamlit dashboard allows a user to enter basic flight information and receive an analysis based on the reference data.

## Project Overview

Drone flight datasets can contain a large number of telemetry and trajectory observations. Analysing these records manually can make it difficult to identify unusual movement patterns.

AeroVista provides a simple decision-support dashboard where the user enters:

- Distance travelled
- Flight duration
- Maximum altitude

The application then derives analytical information from these values and compares the entered flight with patterns learned from the reference UAV dataset.

The output includes:

- Estimated flight speed
- Behaviour group
- Altitude information
- Comparison with reference data
- Potentially unusual flight indication
- Business Intelligence insights

## Main Objective

The main objective of AeroVista is to demonstrate how **Data Mining and Business Intelligence can be applied to UAV flight data**.

The system aims to:

1. Prepare and analyse UAV trajectory data.
2. Identify different UAV movement behaviour groups.
3. Detect observations that differ from reference patterns.
4. Allow a user to enter basic flight information.
5. Compare the user's flight with the reference dataset.
6. Present the results through an interactive BI dashboard.

## Dataset

The project uses UAV data processed from two sources.

### Source 1 — Real UAV Telemetry

This source contains approximately **785 telemetry files** with UAV flight information.

The data includes telemetry-related fields such as:

- Timestamp
- Drone ID
- Mission status
- Position and flight information

### Source 2 — UAV Trajectory Dataset

The trajectory source contains **150,000 records** representing UAV movement using positional information.

Important fields include:

- Timestamp
- X position
- Y position
- Z position

The X, Y and Z values represent the UAV's 3D position.

### Combined Dataset

The processed sources were combined into the project's final dataset containing approximately:

**150,785 records**

The project performs data cleaning and preprocessing before analysis.

## Data Mining Techniques

### K-Means Clustering

K-Means clustering is used to divide the UAV observations into **three behaviour groups** based on their movement characteristics.

The groups are discovered from the data rather than manually assigned.

The exact characteristics of each group are determined by the movement features of the observations assigned to that cluster.

### Anomaly Detection

The project also identifies observations that differ from the reference UAV movement patterns.

These observations are treated as **potentially unusual** and can be selected for further inspection.

A potentially unusual result does not automatically mean that a flight is unsafe. It indicates that the entered behaviour differs from the reference patterns.

## Business Intelligence

AeroVista uses **descriptive and diagnostic Business Intelligence**.

The dashboard answers questions such as:

- What are the characteristics of the entered flight?
- Which behaviour group is it closest to?
- How does it compare with the reference UAV data?
- Does its behaviour differ from the reference patterns?
- Does the flight require further review?

### BI Features

The dashboard provides:

- KPI cards
- Reference dataset statistics
- Flight analysis results
- Behaviour group classification
- Reference-data comparison
- Business Intelligence insights
- Interactive user inputs
- Visual comparison charts

## User Interaction

The user does not need to understand raw UAV telemetry.

The dashboard asks for only basic flight information:

**Distance travelled → Flight duration → Maximum altitude**

After entering the values, the user clicks:

**ANALYSE FLIGHT**

AeroVista processes the input and displays the corresponding analysis using patterns learned from the reference dataset.

## Technology Used

- **Python** — Main programming language
- **Pandas** — Data processing and analysis
- **NumPy** — Numerical operations
- **Scikit-learn** — Data mining and machine-learning algorithms
- **K-Means** — Behaviour clustering
- **Anomaly Detection** — Identification of unusual observations
- **Streamlit** — Interactive web dashboard
- **Plotly** — Interactive data visualisation

## System Workflow

```text
UAV Data Sources
       ↓
Data Collection
       ↓
Data Cleaning & Preprocessing
       ↓
Feature Engineering
       ↓
Exploratory Data Analysis
       ↓
K-Means Behaviour Clustering
       ↓
Anomaly Detection
       ↓
Reference Behaviour Analysis
       ↓
Streamlit BI Dashboard
       ↓
User Flight Input
       ↓
Comparison with Reference Patterns
       ↓
Flight Analysis & BI Insights
