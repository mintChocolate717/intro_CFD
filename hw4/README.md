# COE 347 - Computational Fluid Dynamics  
## Homework 4  

### **Course Information**  
- **Course:** COE 347 – Computational Fluid Dynamics  
- **Institution:** University of Texas at Austin  
- **Instructor:** F. Bisetti  

### **Objective**  
This homework applies **finite difference methods** to solve a **1D Poisson equation** numerically. The assignment involves:  
1. Deriving finite difference approximations.  
2. Implementing numerical solvers for **boundary value problems (BVPs)**.  
3. Analyzing numerical errors and their convergence rates.  

### **Files in this Directory**  
- `hw4.ipynb` - Jupyter Notebook containing code implementations, analysis, and results.  
- `hw4.pdf` - Homework assignment description.  
- `solutionA_N10000.dat` - Reference numerical solution for **Dirichlet BC case** ($u(0) = 0$, $u(1) = 2$).  
- `solutionB_N10000.dat` - Reference numerical solution for **Mixed BC case** ($u'(0) = 10$, $u(1) = 2$).  

### **Tasks Overview**  
1. **Finite Difference Approximation (20 pts)**  
   - Use **Taylor series expansion** to derive the second-order finite difference formula.  
   - Find the constant **C** in the truncation error term.  

2. **Numerical Solution using Second-Order FDF (20 pts)**  
   - Implement a solver for the **Dirichlet boundary conditions** problem.  
   - Compare the numerical solution with **solutionA_N10000.dat**.  
   - Output solution in tabular and graphical form.  

3. **Error Analysis & Convergence Study (20 pts)**  
   - Compute global ($E$) and local ($e$) errors for varying **grid resolutions ($N$)**.  
   - Produce a **log-log plot** of error vs. grid spacing ($1/h$).  
   - Fit errors to **$O(h^\alpha)$** and extract **$\alpha$ values**.  

4. **One-Sided Finite Difference Formula (20 pts)**  
   - Derive a second-order **one-sided approximation** for **Neumann BC ($u'(0) = 10$)**.  
   - Solve the BVP with this boundary condition and compare with **solutionB_N10000.dat**.  

5. **Numerical Solution with Mixed Boundary Conditions (20 pts)**  
   - Implement the solver for **$u'(0) = 10$, $u(1) = 2$** using second-order FDF.  
   - Compare with **solutionB_N10000.dat**.  

6. **Extra Credit: First-Order vs. Second-Order Neumann BC (20 pts)**  
   - Modify solver to use a **first-order** approximation for **$u'(0) = 10$**.  
   - Compute and analyze errors for both **first-order and second-order Neumann BCs**.  
   - Produce **log-log plots** comparing error convergence rates.  

### **Usage Instructions**  
- Run `hw4.ipynb` to generate numerical solutions and plots.  
- Ensure the provided reference solutions (`solutionA_N10000.dat` and `solutionB_N10000.dat`) are in the same directory.  
- The scripts include error computation and visualization tools.  

### **Expected Results & Observations**  
- The second-order scheme provides **better accuracy and faster error convergence**.  
- The global error for **first-order Neumann BC** converges **slower** than the second-order scheme.  
- The log-log plots reveal **order of accuracy differences** in boundary handling techniques.  

### **References**  
- Computational Fluid Dynamics (COE 347 Lecture Notes)  
- Finite Difference Methods for Differential Equations  
