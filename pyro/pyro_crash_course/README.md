# 🔥 Pyro Crash Course

Welcome to the **Atlas Optimization Pyro Crash Course**!  
In this quick, hands-on series, you'll learn how to:

- Build probabilistic models (linear, hierarchical, and nonlinear)
- Fit models using Stochastic Variational Inference (SVI)
- Handle discrete latent variables
- Incorporate neural networks into probabilistic workflows

We illustrate all concepts using a simple **thermal sensor calibration** problem – no prior physics knowledge required.

---

## How to Start 🚀

1. Watch the companion videos on [youtube](https://youtube.com) if you like more context
2. Open the first sprint notebook here:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atlas-optimization/tutorials/blob/main/pyro/pyro_crash_course/pyro_cc_1_minimal_inference.ipynb)

3. Run the installation & data generation cells.
4. Dive into modeling, inference, and critical thinking!
5. Rinse and repeat with examples of increasing complexity.

---

## 📚 Course Structure

Each notebook corresponds to a stage in model complexity:

| Notebook                       | Description                                      |
|-------------------------------|--------------------------------------------------|
| `cc_0_minimal_inference`      | Minimal example: fit parameters using SVI        |
| `cc_1_hello_dataset`          | Generate and explore synthetic sensor data       |
| `cc_2_model_0`                | Model with no trainable parameters               |
| `cc_2_model_1`                | Model with deterministic (fitted) parameters     |
| `cc_2_model_2`                | Add latent variables for hidden causes           |
| `cc_2_model_3`                | Add hierarchical (group-level) randomness        |
| `cc_2_model_4`                | Handle discrete latent variables (e.g. failure)  |
| `cc_2_model_5`                | Incorporate a neural net for nonlinear effects   |

---


## 🧭 Modeling Philosophy

Pyro encourages a different way of thinking about modeling:

> ❌ Traditional Data processing: Design and apply yourself a sequence of **operations** to transform data into meaningful outputs.
>
> ✅ Probabilistic modeling: Specify a **generative process** for data and then let pyro perform inference automatically.

You declare the *model* and the *goal* (e.g. fit latent variables, maximize likelihood), and the inference machinery is handled under the hood automatically.

---

```
*This series is created for educational purposes by Dr. Jemil Avers Butt, Atlas Optimization GmbH – [www.atlasoptimization.com](https://www.atlasoptimization.com).*
```





