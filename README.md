# German Credit Multi-Objective Optimization

This project focuses on applying multi-objective optimization techniques to the German Credit dataset. The primary objectives are to enhance classification performance while simultaneously addressing fairness concerns. Leveraging tools like Optuna for hyperparameter tuning, the project aims to balance accuracy with fairness metrics in the form  of Demographic Parity Difference(DPD) Score.

## Repository Structure

The repository is organized as follows:

```
German_Credit_Multi_Objective/
├── data/
│   └── ...
├── plots/
│   └── ...
├── .gitignore
├── baseline.ipynb
├── optuna-solver.ipynb
├── pre-process.ipynb
├── README.md
├── Report.pdf
├── requirements.txt
```



* **data/**: Contains the German Credit dataset and any additional data files.
* **plots/**: Stores generated plots and visualizations from the analysis.
* **.gitignore**: Specifies files and directories to be ignored by Git.
* **baseline.ipynb**: Establishes baseline models and performance metrics.
* **optuna-solver.ipynb**: Implements Optuna for hyperparameter optimization.
* **pre-process.ipynb**: Handles data cleaning, encoding, and preprocessing steps.

## Getting Started

### Prerequisites

Ensure you have the following installed:

* Python 3.7 or higher
* Jupyter Notebook

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/AyushCodez/German_Credit_Multi_Objective.git
   cd German_Credit_Multi_Objective
   ```



2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```



3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preprocessing:**

   Open and run `pre-process.ipynb` to clean and preprocess the dataset. This will prepare the data for modeling.

2. **Baseline Modeling:**

   Run `baseline.ipynb` to establish baseline performance metrics using standard classification algorithms.

3. **Hyperparameter Optimization:**

   Execute `optuna-solver_seeded.ipynb` to perform hyperparameter tuning using Optuna. This notebook aims to optimize model performance while considering fairness metrics.

## Results

Generated plots and results from the analysis can be found in the `plots/` directory. These visualizations help in understanding the trade-offs between different objectives and the performance of various models.

![Pareto Front](plots/Pareto%20Front%20%20DPD%20vs%20Accuracy.png)
![Pareto front](plots/Pareto%20Solutions%20DPD%20VS%20Accuracy.png)

## Report
For further details regarding the motivation and implementation, please refer to [Report](Report.pdf)



