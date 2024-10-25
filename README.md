Here's the English translation of your machine learning lab README:

---

# Machine Learning Labs

[![GitHub stars](https://img.shields.io/github/stars/HugoPhi/MachineLearningLabs.svg?style=social)](https://github.com/HugoPhi/MachineLearningLabs/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/HugoPhi/MachineLearningLabs.svg?style=social)](https://github.com/HugoPhi/MachineLearningLabs/network/members)
[![GitHub license](https://img.shields.io/github/license/HugoPhi/MachineLearningLabs.svg)](https://github.com/HugoPhi/MachineLearningLabs/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/HugoPhi/MachineLearningLabs.svg)](https://github.com/HugoPhi/MachineLearningLabs/issues)

[中文文档](README.md)

---

This repository contains machine learning models implemented from scratch using `numpy`, `pandas`, and `matplotlib`, aimed at helping learners understand the internal workings of various machine learning algorithms. If you encounter any issues while using this repository, feel free to open an [Issue](https://github.com/HugoPhi/MachineLearningLabs/issues) or improve the project by submitting a Pull Request.

## Features

- Implement machine learning algorithms manually and package them in Python for testing.
- Provide comprehensive basic tests.
- Easily create and run custom tests.

## Install Dependencies

Use the `requirements.txt` file to configure the environment dependencies. It’s recommended to use [pload](https://github.com/HugoPhi/python_venv_loader) as a lightweight virtual environment management tool focused on managing Python environments. To set it up, follow these steps:

```bash
pload new -m 'MLLabs' -v 3.8.20 -f requirements.txt  # v3.8.20 recommended; other versions not tested.
```

## Usage

The following sections outline how to run benchmark tests, custom models, and create and execute tests.

### 1. Run Benchmark Tests

First, create a dedicated virtual environment for the project and activate it. To learn more about using pload to create virtual environments, refer to: [Creating Virtual Environments with pload](https://github.com/HugoPhi/python_venv_loader).

1. **Clone the Repository:**

    ```bash
    git clone --branch main --single-branch https://github.com/HugoPhi/MachineLearningLabs.git
    ```

2. **Install Local Library:**

    Enter the project directory:

    ```bash
    cd MachineLearningLabs/
    ```

    Compile and install the library:

    ```bash
    pip install .
    ```

    You can verify successful installation by running `pip list` and checking if the `hym` library is included.

3. **Run Tests:**

    For example, to run the `test/DecisionTree/watermelon2.0` experiment, execute the following in the project directory:

    ```bash
    python ./test/DecisionTree/watermelon2.0/main.py
    ```

    This will generate the experiment results.

### 2. Custom Models

You can modify or create your own machine learning models. The project structure is divided into two main parts: `src` and `test`. The `src` directory stores the source code for machine learning algorithms, while the `test` directory stores basic and custom tests for each algorithm. Understanding the project structure will help you modify it more efficiently.

#### 2.1 `src` Directory

The `src` directory stores the source code and is organized as follows:

```
src/
├── hym/
│   ├── __init__.py
│   ├── DecisionTree/
│   │   ├── __init__.py
│   │   ├── DecisionTree.py
│   │   └── ...
│   ├── LinearRegression/
│   └── ...
```

- **`hym/`**: Top-level module containing the implementations of various machine learning algorithms.
- To add a new algorithm category, such as Support Vector Machine, create a `SupportVectorMachine/` directory under `hym/`, and add it to `hym/__init__.py` as follows:

    ```python
    from . import DecisionTree
    from . import LinearRegression
    from . import SupportVectorMachine  # New algorithm module

    __all__ = [
        'DecisionTree',
        'LinearRegression',
        'SupportVectorMachine'  # Add new module
    ]
    ```

- **File Naming Conventions:**

    1. **Algorithm Class Files**: Use CamelCase, e.g., `BasicDecisionTree.py`, `Variants.py`, for implementing algorithm classes.
    2. **Helper Class Files**: Use snake_case, e.g., `node.py`, for implementing helper classes.
    3. **Helper Function Files**: `utils.py` contains utility functions, e.g., data loading, preprocessing, and math functions.
    4. **Package Initialization File**: `__init__.py`, which marks the package and submodules, with exported contents listed in `__all__`.

#### 2.2 `test` Directory

The `test` directory stores test code and is structured similarly to `src`:

```
test/
├── DecisionTree/
│   ├── iris/
│   │   ├── iris.xlsx
│   │   └── main.py
│   ├── watermelon2.0/
│   │   ├── watermelon2.0.xlsx
│   │   └── main.py
│   └── ...
├── LinearRegression/
└── ...
```

- Create directories under `test/` by algorithm category, matching the structure in `src/`.
- Inside each algorithm directory, add test cases. Some tests for basic datasets are already provided, but you can also add your own experiments.

#### 2.3 Other Important Files

1. **setup.py**

    Contains package build information such as version, dependencies, and author information. The version format follows `v[x].[y].[z]`, where:

    - `x`: Major updates, breaking API changes.
    - `y`: Significant new features, such as implementing a new algorithm category.
    - `z`: Minor updates, including bug fixes or small adjustments.

2. **README.md**

    Documents usage and updates. It’s recommended to check periodically for the latest information.

## Progress

<details>
<summary>Algorithm Library</summary>

- [ ] **Supervised Learning**
  - [ ] Linear Regression
  - [x] Logistic Regression
  - [x] Decision Tree
    - [x] ID3
    - [x] C4.5
    - [ ] CART
  - [ ] Support Vector Machine
  - [ ] Neural Networks
- [ ] **Unsupervised Learning**
  - [ ] K-means Clustering
  - [ ] Principal Component Analysis
     
</details>

<details>
<summary>Testing</summary>

- [ ] **Supervised Learning**
  - [ ] Linear Regression
  - [x] Logistic Regression
    - [x] iris
  - [x] Decision Tree
    - [x] watermelon2.0
    - [x] iris
    - [ ] ice-cream
    - [x] wine quality
    - [ ] house price
  - [ ] Support Vector Machine
  - [ ] Neural Networks
- [ ] **Unsupervised Learning**
  - [ ] K-means Clustering
  - [ ] Principal Component Analysis

</details>

## References

Add your references here.

## License

This project is licensed under the [MIT License](LICENSE). Please refer to the LICENSE file for details.

---

If you find this project helpful, please consider [⭐️ starring](https://github.com/HugoPhi/MachineLearningLabs) us!

[![Star History Chart](https://api.star-history.com/svg?repos=HugoPhi/MachineLearningLabs&type=Timeline)](https://star-history.com/#HugoPhi/MachineLearningLabs&Timeline)

---
