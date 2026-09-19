# Corporate Heist - Solution Documentation

**Team Name:** NeuralKnights  
**Members:** Rajnish Kumar, Prince Kumar  
**Event:** Innov8 4.0 Hackathon (Eightfold AI X Aries IIT Delhi)

## Overview
Our candidate shortlisting pipeline is engineered using HistGradientBoosting model to accurately score and rank candidates for "The Corporate Heist". The solution dynamically aligns feature spaces between historical training data and current test data to prevent structural drift.

## Repository Contents
- `solution.py`: Model logic and prediction script.
- `check_format.py`: Output structure validation script.
- `documentation.pdf`: Detailed methodology and solution write-up.
- `Corporate_Heist_Problem_Statement.pdf`: Hackathon problem brief.

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas scikit-learn
