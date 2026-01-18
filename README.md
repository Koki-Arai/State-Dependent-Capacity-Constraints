# State-Dependent Capacity Constraints: Evidence from Japanese Procurement Auctions

This repository contains the replication code and data documentation for the paper **"State-Dependent Capacity Constraints: Evidence from Japanese Procurement Auctions"**.

The project estimates a dynamic auction model using Japanese public procurement data (2017-2024). It combines reduced-form analysis with the structural estimation method of Bajari, Benkard, and Levin (2007) to quantify the shadow cost of capacity constraints and identify boundary conditions for dynamic auction theory.

## Repository Structure

* `data/`: Directory for input datasets (see Data Availability below).
* `src/`: Python source code for analysis.
    * `replication_main.py`: Main script to reproduce all tables and figures.
* `output/`: Directory where results (tables, regression summaries) are saved.
* `requirements.txt`: List of Python dependencies.

## Key Features

1.  **Dynamic Backlog Calculation**: Algorithms to reconstruct firm-level backlog from contract history.
2.  **Reduced-Form Analysis**: OLS and Logistic regressions identifying the "bidding puzzle" (higher backlog -> higher bids but higher win probability).
3.  **Structural Estimation (BBL 2007)**: Two-step inequality estimator to recover the shadow cost of capacity ($\theta_{backlog}$).
4.  **Mechanism Checks**: Verification of boundary conditions (Invisible Backlog, Scoring Bias, Incumbency).

## Installation & Usage

### Prerequisites
* Python 3.8 or higher

### Setup
1.  Clone this repository:
    ```bash
    git clone [https://github.com/Koki-Arai/State-Dependent-Capacity-Constraints.git](https://github.com/Koki-Arai/State-Dependent-Capacity-Constraints.git)
    cd State-Dependent-Capacity-Constraints
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Analysis
To replicate the full analysis (Reduced-form -> Structural -> Counterfactuals), run the main script:

```bash
python src/replication_main.py
