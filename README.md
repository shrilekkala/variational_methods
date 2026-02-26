# Variational Methods (Reading Project): Multi-Block ADMM Convergence

This repository contains a final reading project for a Variational Methods / Optimization course.  
Topic: **convergence behavior of the direct extension of ADMM to multi-block (3+ block) convex minimization**.

**Final deliverables**
- **Report (PDF):** [ADMM Report.pdf](Final%20Project/ADMM%20Report.pdf)
- **Slides (PDF):** [ADMM_Presentation.pdf](Final%20Project/ADMM_Presentation.pdf)

---

## Summary

The classic ADMM (Alternating Direction Method of Multipliers) has well-known convergence guarantees for **two blocks**, but the naive “direct extension” to **three blocks** can fail.

Key takeaways covered in the report/presentation:
- **Sufficient conditions for convergence** exist (e.g., certain **orthogonality** relationships between constraint matrices).
- Without these conditions, the **direct 3-block ADMM can diverge**, even on simple instances.
- The method can have **ergodic convergence rate O(1/k)** (when the assumptions hold).
- **Strong convexity alone does not guarantee convergence** for all penalty parameters.

---

## Citation

This project is based on:
> Caihua Chen, Bingsheng He, Yinyu Ye, Xiaoming Yuan (2016).  
> *The direct extension of ADMM for multi-block convex minimization problems is not necessarily convergent.*
