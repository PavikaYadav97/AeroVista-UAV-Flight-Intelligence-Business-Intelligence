---
license: apache-2.0
language:
- en
tags:
- UAV
- Drones
- Trajectory
pretty_name: U
size_categories:
- 100K<n<1M
---
# UAV Trajectory Dataset

## Summary

This dataset comprises over 5000 random UAV (Unmanned Aerial Vehicle) trajectories collected over 20 hours of flight time. It is intended for training AI models such as trajectory prediction applications. The dataset is generated through an automated pipeline for the creation and preprocessing of UAV synthetic trajectories, making it ready for direct AI model training.

## Data Description

The dataset features parameterized trajectories following predefined patterns, specifically circular and infinity-like paths. 

## Dataset Structure

### Data Fields

- `timestamp`: Recording time of the data point.
- `position`: 3D position of the UAV (x, y, z coordinates).

## Citation

If you use our work in your research, please cite:

Nacar, O.; Abdelkader, M.; Ghouti, L.; Gabr, K.; Al-Batati, A.; Koubaa, A. **VECTOR: Velocity-Enhanced GRU Neural Network for Real-Time 3D UAV Trajectory Prediction**. *Drones* 2025, **9**, 8. [https://doi.org/10.3390/drones9010008](https://doi.org/10.3390/drones9010008)
