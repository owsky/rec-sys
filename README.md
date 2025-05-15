# RecSys

Bundle of recommender systems implemented (mostly) from scratch.

## Models

- Collaborative Filtering
  - Matrix Factorization
    - Stochastic Gradient Descent
    - Alternating Least Squares (Python, sequential)
    - Alternating Least Squares (PySpark, parallelized)
  - Neighborhood using either Pearson Correlation or Adjusted Cosine Similarity
    - User-based
    - Item-based 
- Content Based
- Non-Personalized
  - Highest Rated
  - Most Popular