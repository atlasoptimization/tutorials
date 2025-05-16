# 🔥 Pyro Crash Course: Probabilistic Modeling Made Intuitive

Welcome to the **Atlas Optimization Pyro Crash Course**!  
This hands-on tutorial series teaches you how to build, understand, and apply **probabilistic models** using [Pyro](https://pyro.ai) – a powerful PPL built on PyTorch.

We use a **sensor calibration scenario** as a motivating example:  
Imagine you measure temperatures with imperfect thermistors — how do you infer their offset, scale, uncertainty, or failure modes?

👉 [Download `pyro_crash_course` as ZIP](https://download-directory.github.io/?url=https://github.com/atlasoptimization/tutorials/tree/master/pyro/pyro_crash_course)


---

## 🎓 What You Will Learn

- How to **build probabilistic models** from scratch using Pyro primitives  
- How to **infer hidden structure** in data (offsets, noise, failure) using SVI  
- How to **encode model uncertainty** with latent variables and priors  
- How to handle **hierarchical effects** and **discrete latent variables**  
- How to plug in **neural networks** as flexible function approximators in a probabilistic setting

Each concept is introduced in a clean, focused notebook and grounded in a simple real-world task.

---

## 📦 How to Use

1. [Watch the companion videos](https://youtube.com) for walk-throughs and background
2. Launch any notebook below using **Google Colab**  
3. Follow the code + markdown explanation and interact with the models  
4. Compare pre- and post-training predictions, inspect posterior distributions, and reflect on model performance
5. Repeat with growing complexity!

---

## 🗂️ Crash Course Overview

| Notebook | Description | Launch |
|----------|-------------|--------|
| `cc_0_minimal_inference` | Minimal Pyro example: sampling and inference with a single scalar | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atlasoptimization/tutorials/blob/master/pyro/pyro_crash_course/pyro_cc_0_minimal_inference.ipynb) |
| `cc_1_hello_dataset` | Generate and visualize synthetic sensor data | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atlasoptimization/tutorials/blob/master/pyro/pyro_crash_course/pyro_cc_1_hello_dataset.ipynb) |
| `cc_2_model_0` | No free parameters: just generate + visualize predictive samples | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atlasoptimization/tutorials/blob/master/pyro/pyro_crash_course/pyro_cc_2_model_0.ipynb) |
| `cc_3_model_1` | Add deterministic parameters and fit with SVI (like least squares) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atlasoptimization/tutorials/blob/master/pyro/pyro_crash_course/pyro_cc_3_model_1.ipynb) |
| `cc_4_model_2` | Treat parameters as **latent variables** with priors and posterior inference | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atlasoptimization/tutorials/blob/master/pyro/pyro_crash_course/pyro_cc_4_model_2.ipynb) |
| `cc_5_model_3` | Add **hierarchical structure**: each sensor has its own latent parameters | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atlasoptimization/tutorials/blob/master/pyro/pyro_crash_course/pyro_cc_5_model_3.ipynb) |
| `cc_6_model_4` | Introduce **discrete latent variables** for outlier detection | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atlasoptimization/tutorials/blob/master/pyro/pyro_crash_course/pyro_cc_6_model_4.ipynb) |
| `cc_7_model_5` | Use a **neural net** as part of the generative process for flexible modeling | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atlasoptimization/tutorials/blob/master/pyro/pyro_crash_course/pyro_cc_7_model_5.ipynb) |

---

## 🧭 Modeling Philosophy

Pyro encourages a different way of thinking about modeling:

> ❌ Traditional Data processing: Design and apply yourself a sequence of **operations** to transform data into meaningful outputs.
>
> ✅ Probabilistic modeling: Specify a **generative process** for data and then let pyro perform inference automatically.

You declare the *model* and the *goal* (e.g. fit latent variables, maximize likelihood), and the inference machinery is handled under the hood automatically.



---

## 🧠 Math and Modeling Concepts at a Glance

Each notebook teaches essential probabilistic programming ideas:

| Concept | Introduced In | Description |
|--------|----------------|-------------|
| **ELBO / Variational Inference** | `cc_3_model_1` onward | Fit posterior by maximizing evidence lower bound |
| **Latent Variables** | `cc_4_model_2` | Treat unknowns as random variables with priors |
| **Hierarchical Modeling** | `cc_5_model_3` | Pool information across multiple sensors |
| **Discrete Latents & Enumeration** | `cc_6_model_4` | Use latent classes (e.g. faulty vs. healthy) |
| **Amortized Inference / Neural Likelihood** | `cc_7_model_5` | Neural nets within probabilistic models |

---

## 🔎 Why It Matters

Classical models assume known structure and known parameters.  
**Probabilistic programming lets you invert** the process:  

📌 You **specify the data generation process**,  
and Pyro **infers what must have happened** given your observations.

This is crucial for:

- Uncertainty quantification  
- Fault detection & robustness  
- Scientific modeling with hidden structure  
- Simulation-based inference & data-driven physics

---

*This series is created by Dr. Jemil Avers Butt, Atlas Optimization GmbH – [www.atlasoptimization.com](https://www.atlasoptimization.com)*





